"""VLM の指示で四足ロボット(Go2)が荷物(赤い箱)まで歩く。

学習済みの歩行方策(logs/go2-walking)を使い、VLM(Gemini)が「前進/左/右/停止」を選ぶ。
go2_export.zip を解凍したフォルダで実行する:

    python vlm_walk.py --mock        # Gemini なしの動作確認(位置から計算した判断を使う)
    python vlm_walk.py               # Gemini を使う(環境変数 GEMINI_API_KEY が必要)

フレームは frames_vlm/ に保存し、ffmpeg で vlm_walk.mp4 にする。
"""
import argparse
import io
import json
import math
import os
import pathlib
import pickle
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())

import imageio
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# ---------------------------------------------------------------- 設定
INSTRUCTION = "赤い荷物のところまで歩いて、近くまで来たら止まって"
RED_POS = (3.0, 0.6)      # 荷物(赤い箱)の位置
BLUE_POS = (-1.5, 3.0)    # 邪魔者(青い箱)の位置
START_YAW_DEG = 90.0      # 最初は +y 方向を向く(赤い箱は右手に見える)
DECIDE_EVERY = 40         # 何ステップごとに VLM に聞くか (50Hz なので 0.8 秒)
MAX_STEPS = 1500
HOLD_AFTER_STOP = 60
CMD = {                   # action -> (前進速度, 横速度, 旋回速度)
    "forward": (0.5, 0.0, 0.0),
    "left": (0.1, 0.0, 0.6),
    "right": (0.1, 0.0, -0.6),
    "stop": (0.0, 0.0, 0.0),
}
ACTIONS = list(CMD)

PROMPT = f"""あなたは四足歩行ロボットの操縦を担当しています。
画像はロボットの真後ろ少し上から、ロボットの進行方向を見たものです。画像の中央がロボットの正面です。
指示: 「{INSTRUCTION}」
次の4つから行動を1つだけ選んでください。
- forward: 赤い箱が画面の中央付近(ロボットの真正面)にあり、まだロボットとの間に床が見えているとき
- left: 赤い箱が画面の左寄りにあるとき、または赤い箱が見えないとき(左に回って探す)
- right: 赤い箱が画面の右寄りにあるとき
- stop: 赤い箱の下の端が、ロボットの体(背中や頭)の上の端に重なる、接する、または隣り合うように見えたとき。
  これは箱がロボットのすぐ目の前にある合図です。箱が小さく見えていても、ロボットと重なっていれば stop にしてください。
青い箱は目的の荷物ではありません。reason には判断の理由を日本語で20文字以内で書いてください。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}


# ---------------------------------------------------------------- go2_env.py への追加
def patch_env_file():
    f = pathlib.Path(os.getcwd()) / "go2_env.py"
    src = f.read_text(encoding="utf-8")
    build = "self.scene.build(n_envs=num_envs)"
    assert build in src, "go2_env.py の構成が想定と違います"
    if "self.cam" not in src:
        cam = "self.cam = self.scene.add_camera(res=(960, 540), pos=(3.0, -2.0, 1.5), lookat=(0, 0, 0.3), fov=70, GUI=False)"
        src = src.replace(build, cam + "\n        " + build)
    src = src.replace("fov=40", "fov=70")
    if "self.box_red" not in src:
        boxes = (
            "self.box_red = self.scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(%s, %s, 0.2), fixed=True),"
            " surface=gs.surfaces.Plastic(color=(0.9, 0.1, 0.1)))\n        "
            "self.box_blue = self.scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(%s, %s, 0.2), fixed=True),"
            " surface=gs.surfaces.Plastic(color=(0.1, 0.2, 0.9)))\n        " % (RED_POS + BLUE_POS)
        )
        src = src.replace("self.cam =", boxes + "self.cam =", 1)
    f.write_text(src, encoding="utf-8")


# ---------------------------------------------------------------- VLM
def make_vlm(args):
    if args.mock:
        def decide_mock(frame, state):
            dx, dy = RED_POS[0] - state["x"], RED_POS[1] - state["y"]
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx) - state["yaw"]
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            if dist < 0.9:
                return {"action": "stop", "reason": "(mock) 目の前に到着"}
            if abs(ang) < 0.25:
                return {"action": "forward", "reason": "(mock) 正面にある"}
            return {"action": "left" if ang > 0 else "right", "reason": "(mock) 向きを合わせる"}
        return decide_mock

    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("環境変数 GEMINI_API_KEY が設定されていません。--mock なら API キー無しで動かせます。")
    client = genai.Client(api_key=key)

    # シミュレーションは呼び出し中は止まっているので、待っても動画には影響しない。
    # 無料枠の回数制限を避けるため、呼び出しの間隔をあける。
    # --model は "モデルA,モデルB" のように複数指定できる。1日の上限に達したら次のモデルに切り替える。
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    mem = {"last_call": 0.0, "last": None, "fails": 0, "mi": 0}

    def decide_gemini(frame, state):
        buf = io.BytesIO()
        Image.fromarray(frame).convert("RGB").save(buf, format="JPEG", quality=85)
        last_err = None
        for attempt in range(5):
            if mem["mi"] >= len(models):
                break
            wait = args.min_interval - (time.time() - mem["last_call"])
            if wait > 0:
                time.sleep(wait)
            mem["last_call"] = time.time()
            try:
                resp = client.models.generate_content(
                    model=models[mem["mi"]],
                    contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=SCHEMA, temperature=0.0),
                )
                out = json.loads(resp.text)
                if out.get("action") in CMD:
                    mem["last"], mem["fails"] = out, 0
                    return out
                last_err = f"想定外の応答: {resp.text}"
            except Exception as e:
                last_err = e
            print(f"  VLM 呼び出し失敗({attempt + 1}/5): {str(last_err)[:300]}", flush=True)
            with open("vlm_errors.log", "a", encoding="utf-8") as f:
                f.write(f"step {state['step']} attempt {attempt + 1}: {last_err}\n")
            if "PerDay" in str(last_err) or "NOT_FOUND" in str(last_err):  # 待っても直らない: 次のモデルへ
                print(f"  モデル {models[mem['mi']]} は使えません(1日の上限、または存在しない)", flush=True)
                mem["mi"] += 1
                if mem["mi"] < len(models):
                    print(f"  → {models[mem['mi']]} に切り替えます", flush=True)
                continue
            time.sleep(min(60, 5 * 2 ** attempt))  # 5,10,20,40,60 秒と待つ
        mem["fails"] += 1
        if mem["last"] is None or mem["fails"] >= 3 or mem["mi"] >= len(models):
            return {"action": "stop", "reason": "VLM が使えないため停止"}
        return {"action": mem["last"]["action"], "reason": "(VLM エラー: 前の行動を継続)"}
    return decide_gemini


# ---------------------------------------------------------------- 表示用
def load_font(size):
    for name in ["meiryo.ttc", "YuGothM.ttc", "msgothic.ttc", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = load_font(26)
FONT_S = load_font(20)


def overlay(frame, step, decision):
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, img.width, 92], fill=(0, 0, 0, 150))
    d.text((14, 8), f"指示: {INSTRUCTION}", font=FONT, fill=(255, 255, 255))
    if decision is not None:
        d.text((14, 48), f"VLM: {decision['action']}  / {decision['reason']}", font=FONT_S, fill=(255, 230, 120))
    d.text((img.width - 150, img.height - 34), f"t = {step / 50:5.1f} s", font=FONT_S, fill=(255, 255, 255))
    return img


# ---------------------------------------------------------------- メイン
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--min-interval", type=float, default=7.0, help="VLM 呼び出しの最小間隔(秒)")
    ap.add_argument("--out", default="frames_vlm")
    args = ap.parse_args()

    patch_env_file()
    from go2_env import Go2Env

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = "logs/go2-walking"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["resampling_time_s"] = 1e9  # 指令は VLM が決める(自動の再抽選を止める)
    env_cfg["episode_length_s"] = 600.0  # 標準は20秒で時間切れリセットになるため延ばす
    half = math.radians(START_YAW_DEG) / 2
    env_cfg["base_init_quat"] = [math.cos(half), 0.0, 0.0, math.sin(half)]

    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                 command_cfg=command_cfg, show_viewer=False)

    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt = max(int(f.split("_")[1].split(".")[0]) for f in os.listdir(log_dir) if f.startswith("model_"))
    runner.load(f"{log_dir}/model_{ckpt}.pt")
    policy = runner.get_inference_policy(device=gs.device)

    decide = make_vlm(args)
    cam = env.cam
    out_dir = pathlib.Path(args.out)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()

    obs = env.reset()
    decision, log, stop_at, n = None, [], None, 0
    with torch.no_grad():
        for step in range(MAX_STEPS):
            p = env.base_pos[0].cpu().numpy()
            w, x, y, z = [float(v) for v in env.base_quat[0].cpu().numpy()]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            fwd = (math.cos(yaw), math.sin(yaw))
            cam.set_pose(pos=(p[0] - 2.2 * fwd[0], p[1] - 2.2 * fwd[1], 1.3),
                         lookat=(p[0] + 1.5 * fwd[0], p[1] + 1.5 * fwd[1], 0.2), up=(0.0, 0.0, 1.0))
            frame = cam.render()[0]

            if step % DECIDE_EVERY == 0 and stop_at is None:
                state = {"step": step, "x": float(p[0]), "y": float(p[1]), "yaw": yaw}
                decision = decide(frame, state)
                log.append({"step": step, **state, **decision})
                print(f"step {step:4d} ({step / 50:5.1f}s)  {decision['action']:7s} {decision['reason']}", flush=True)
                if decision["action"] == "stop":
                    stop_at = step
            if stop_at is not None and step - stop_at > HOLD_AFTER_STOP:
                break

            vx, vy, wz = CMD[decision["action"]]
            env.commands[:] = torch.tensor([[vx, vy, wz]], dtype=gs.tc_float, device=gs.device)
            overlay(frame, step, decision).save(out_dir / f"{n:04d}.png")
            n += 1
            obs, _, dones, _ = env.step(policy(obs))
            if bool(dones[0]):
                print("転倒などでリセットされたため終了")
                break

    json.dump(log, open("vlm_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n} フレームを {out_dir}/ に保存")

    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "50", "-i", f"{out_dir}/%04d.png",
                        "-pix_fmt", "yuv420p", "vlm_walk.mp4"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("vlm_walk.mp4 を作成しました")
    except Exception as e:
        print("mp4 の作成に失敗:", e)
        print(f"手動: ffmpeg -y -framerate 50 -i {out_dir}/%04d.png -pix_fmt yuv420p vlm_walk.mp4")


if __name__ == "__main__":
    main()
