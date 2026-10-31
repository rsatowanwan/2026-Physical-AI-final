"""複数シーンで VLM(Gemini) が Go2 を誘導し、1本の動画にまとめる。

vlm_walk.py と同じフォルダで実行する(go2_export.zip を解凍した C:\\work\\go2 など)。

    python vlm_scenes.py --mock --scenes A,B,C,D     # Gemini なしの動作確認
    python vlm_scenes.py --scenes A,B --model gemini-3.5-flash
    python vlm_scenes.py --assemble                   # 保存済みシーンをつないで final.mp4 を作る

シーンごとに scene_A/ などへフレームとログを保存するので、無料枠の都合で日を分けて実行できる。
"""
import argparse
import json
import math
import os
import pathlib
import pickle
import shutil
import subprocess
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
from PIL import Image, ImageDraw
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
import vlm_walk as W

COLOR_JA = {"red": "赤", "blue": "青"}
BOX = {"red": W.RED_POS, "blue": W.BLUE_POS}

I_RED = "赤い荷物のところまで歩いて、近くまで来たら止まって"
I_BLUE = "青い荷物のところまで歩いて、近くまで来たら止まって"
I_BLUE_NEXT = "次は、青い荷物のところまで歩いて、近くまで来たら止まって"

# start = (x, y, 向き[度])  向き 0=+x方向, 90=+y方向
SCENES = {
    "A": {"title": "シーン1  赤い荷物へ",
          "note": "赤い箱が見えている状態から、指示どおりに近づく",
          "start": (0.0, 0.0, 90.0), "phases": [(I_RED, "red")]},
    "B": {"title": "シーン2  指示を変える → 青い荷物へ",
          "note": "同じ場所・同じ景色で、指示の色だけを変える",
          "start": (0.0, 0.0, 90.0), "phases": [(I_BLUE, "blue")]},
    # シーン3は、無料枠の1日の回数に収まるよう C1 と C2 に分けて実行する(動画では1つにつなぐ)
    "C1": {"title": "シーン3  荷物が見えない向きから探す → 指示の切り替え",
           "note": "後ろ向きで開始。見つけて近づいたあと、途中で指示を青に切り替える",
           "start": (0.0, 0.0, 180.0), "phases": [(I_RED, "red")]},
    "C2": {"title": None, "note": "", "start": "continue:C1", "phases": [(I_BLUE_NEXT, "blue")]},
    "D": {"title": "シーン4  目の前の赤い箱を無視して、青い荷物を探す",
          "note": "赤い箱は目的ではない。青い箱は後ろにあるので、回って探す",
          "start": (0.0, 0.0, 0.0), "phases": [(I_BLUE, "blue")]},
}

DECIDE_EVERY = 50   # 1.0秒ごとに VLM に聞く
MAX_PHASE_STEPS = 900
HOLD_AFTER_STOP = 60


def make_prompt(instruction, target):
    t = COLOR_JA[target]
    o = COLOR_JA["blue" if target == "red" else "red"]
    return f"""あなたは四足歩行ロボットの操縦を担当しています。
画像はロボットの真後ろ少し上から、ロボットの進行方向を見たものです。画像の中央がロボットの正面です。
指示: 「{instruction}」
目的の荷物は{t}い箱です。{o}い箱は目的の荷物ではありません。
次の4つから行動を1つだけ選んでください。
- forward: {t}い箱が画面の中央付近(ロボットの真正面)にあり、まだロボットとの間に床が見えているとき
- left: {t}い箱が画面の左寄りにあるとき、または{t}い箱が画面に見えないとき(左に回って探す)
- right: {t}い箱が画面の右寄りにあるとき
- stop: {t}い箱の下の端が、ロボットの背中の一番上の高さに届いた(箱の下の端とロボットの背中の上端が、画面上でほぼ同じ高さになった)とき。
  箱はまだ小さく見えていて、ロボットから少し離れて見えていても、その高さに届いたらすぐに stop にしてください。ぶつかる前に止まることが大切です。
reason には判断の理由を日本語で20文字以内で書いてください。"""


def mock_decider(target):
    def decide(frame, state):
        bx, by = BOX[target]
        dx, dy = bx - state["x"], by - state["y"]
        dist = math.hypot(dx, dy)
        ang = (math.atan2(dy, dx) - state["yaw"] + math.pi) % (2 * math.pi) - math.pi
        if dist < 1.4:  # 箱の下端がロボットの背中の高さに届く距離
            return {"action": "stop", "reason": "(mock) 目の前に到着"}
        if abs(ang) > math.radians(35):
            return {"action": "left" if ang > 0 else "right", "reason": "(mock) 向きを合わせる"}
        return {"action": "forward", "reason": "(mock) 正面にある"}
    return decide


def set_start(env, x, y, yaw_deg):
    from genesis.utils.geom import inv_quat
    half = math.radians(yaw_deg) / 2
    q = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=gs.tc_float, device=gs.device)
    env.init_base_pos = torch.tensor([x, y, env.init_base_pos[2].item()], dtype=gs.tc_float, device=gs.device)
    env.init_base_quat = q
    env.inv_base_init_quat = inv_quat(q)
    env.init_qpos = torch.cat((env.init_base_pos, q, env.init_dof_pos))


def run_scene(name, env, policy, cam, args, vlm_factory):
    sc = SCENES[name]
    out_dir = pathlib.Path(f"scene_{name}")
    failed = False
    start = sc["start"]
    if isinstance(start, str) and start.startswith("continue:"):
        # 前のシーンが止まった位置・向きから続ける
        prev = pathlib.Path(f"scene_{start.split(':')[1]}") / "end_pose.json"
        if not prev.exists():
            sys.exit(f"{prev} がありません。先に {start.split(':')[1]} を成功させてください。")
        pose = json.load(open(prev, encoding="utf-8"))
        start = (pose["x"], pose["y"], math.degrees(pose["yaw"]))
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    set_start(env, *start)
    obs = env.reset()
    n, log = 0, []
    print(f"\n=== シーン {name}: {sc['title'] or '(続き)'} ===", flush=True)
    for pi, (instruction, target) in enumerate(sc["phases"]):
        W.INSTRUCTION = instruction
        W.PROMPT = make_prompt(instruction, target)
        decide = mock_decider(target) if args.mock else vlm_factory()
        decision, stop_at = None, None
        print(f"-- 指示: {instruction}", flush=True)
        with torch.no_grad():
            for step in range(MAX_PHASE_STEPS):
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
                    log.append({"phase": pi, "instruction": instruction, **state, **decision})
                    print(f"  t={step / 50:5.1f}s {decision['action']:7s} {decision['reason']}", flush=True)
                    if decision["action"] == "stop":
                        stop_at = step
                if stop_at is not None and step - stop_at > HOLD_AFTER_STOP:
                    break
                vx, vy, wz = W.CMD[decision["action"]]
                env.commands[:] = torch.tensor([[vx, vy, wz]], dtype=gs.tc_float, device=gs.device)
                W.overlay(frame, n, decision).save(out_dir / f"{n:04d}.png")
                n += 1
                obs, _, dones, _ = env.step(policy(obs))
                if bool(dones[0]):
                    print("  転倒などでリセットされたため、このシーンを終了", flush=True)
                    break
            else:
                print("  時間切れ", flush=True)
        if decision is not None and decision["reason"] == "VLM が使えないため停止":
            failed = True
            break
    json.dump(log, open(out_dir / "log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if failed:
        # 失敗したシーンは、つなぎの動画に混ざらないように別名にしておく
        bad = pathlib.Path(f"scene_{name}_failed")
        shutil.rmtree(bad, ignore_errors=True)
        out_dir.rename(bad)
        print(f"シーン {name}: VLM が使えず途中で止まりました。{bad}/ に退避しました。後でやり直してください。", flush=True)
        return
    p = env.base_pos[0].cpu().numpy()
    w, x, y, z = [float(v) for v in env.base_quat[0].cpu().numpy()]
    json.dump({"x": float(p[0]), "y": float(p[1]), "yaw": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))},
              open(out_dir / "end_pose.json", "w", encoding="utf-8"))
    print(f"シーン {name}: {n} フレーム ({n / 50:.1f} 秒) を {out_dir}/ に保存", flush=True)


# ---------------------------------------------------------------- まとめ(動画の組み立て)
def card(lines, size=(960, 540), sub=None):
    img = Image.new("RGB", size, (14, 22, 34))
    d = ImageDraw.Draw(img)
    y = 90
    for i, line in enumerate(lines):
        font = W.load_font(40 if i == 0 else 28)
        d.text((60, y), line, font=font, fill=(255, 255, 255) if i == 0 else (200, 215, 235))
        y += 70 if i == 0 else 46
    if sub:
        d.text((60, size[1] - 70), sub, font=W.load_font(22), fill=(150, 170, 195))
    return img


def assemble():
    out = pathlib.Path("final_frames")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    n = 0

    def put(img, seconds):
        nonlocal n
        for _ in range(int(seconds * 50)):
            img.save(out / f"{n:04d}.png")
            n += 1

    put(card(["VLM の指示で歩く四足ロボット",
              "",
              "日本語の指示  →  Gemini(VLM)が画像を見て判断",
              "     →  学習済みの歩行方策が足を動かす  →  Go2",
              "",
              "・画面は、VLM に渡している画像そのものです",
              "・歩行は強化学習(Genesis, Go2)、判断は Gemini"],
             sub="Physical AI 応用編1 最終課題"), 6)
    done = []
    for name in SCENES:
        d = pathlib.Path(f"scene_{name}")
        frames = sorted(d.glob("*.png")) if d.exists() else []
        if not frames:
            print(f"シーン {name} は未作成のため飛ばします")
            continue
        sc = SCENES[name]
        if sc["title"]:  # C2 は C1 の続きなので、タイトル画面は入れない
            put(card([sc["title"], "", sc["note"]]), 3)
        for f in frames:
            shutil.copy(f, out / f"{n:04d}.png")
            n += 1
        done.append(name)
    put(card(["まとめ",
              "",
              "・指示の言葉を変えると、目指す荷物が変わる",
              "・荷物が見えなくても、回って探せる",
              "・判断は画像だけ。位置の座標は渡していない"]), 6)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "50", "-i", f"{out}/%04d.png",
                    "-pix_fmt", "yuv420p", "final.mp4"],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"final.mp4 を作成しました ({n / 50:.1f} 秒, シーン: {','.join(done)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="A,B,C,D")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--min-interval", type=float, default=13.0,
                    help="VLM 呼び出しの最小間隔(秒)。無料枠は1分5回までのことがあるので13秒")
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        assemble()
        return

    W.patch_env_file()
    from go2_env import Go2Env

    gs.init(backend=gs.cpu, logging_level="warning")
    log_dir = "logs/go2-walking"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["resampling_time_s"] = 1e9
    env_cfg["episode_length_s"] = 600.0  # 標準は20秒で時間切れリセットになるため延ばす
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                 command_cfg=command_cfg, show_viewer=False)
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt = max(int(f.split("_")[1].split(".")[0]) for f in os.listdir(log_dir) if f.startswith("model_"))
    runner.load(f"{log_dir}/model_{ckpt}.pt")
    policy = runner.get_inference_policy(device=gs.device)

    factory = None if args.mock else (lambda: W.make_vlm(args))
    for name in [s.strip().upper() for s in args.scenes.split(",") if s.strip()]:
        run_scene(name, env, policy, env.cam, args, factory)


if __name__ == "__main__":
    main()
