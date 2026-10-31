"""説明資料(PDF)を作る。

シーンの実行結果(scene_X/log.json と画像)から、表とサムネイルを自動で作り、
HTML にして Microsoft Edge(ヘッドレス)で PDF にする。シーンを実行し直したら、もう一度実行すればよい。

    python report/build_report.py --work-dir <scene_A などがあるフォルダ> --pdf 説明資料_PAI最終課題_<アカウント名>.pdf
"""
import argparse
import html
import json
import math
import pathlib
import shutil
import subprocess
import sys

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
BOX = {"red": (3.0, 0.6), "blue": (-1.5, 3.0)}  # src/vlm_walk.py の RED_POS, BLUE_POS と同じ
JA = {"red": "赤", "blue": "青"}

# シーン: (表示名, 指示, [位相ごとの目的の箱], 開始時の状況)
SCENES = [
    ("A", "赤い荷物へ", ["red"], "赤い箱が見えている"),
    ("B", "青い荷物へ(指示だけ変更)", ["blue"], "Aと同じ場所・景色"),
    ("C1", "後ろ向きから赤い荷物を探す", ["red"], "荷物が視界にない"),
    ("C2", "指示を青に切り替えて青い荷物へ", ["blue"], "C1の止まった位置から続ける"),
    ("D", "赤い箱を無視して青い荷物を探す", ["blue"], "目の前に赤い箱、青は後ろ"),
]


def edge_path():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]:
        if pathlib.Path(p).exists():
            return p
    sys.exit("Microsoft Edge が見つかりません。--edge で msedge.exe のパスを指定してください。")


def summarize(work, name, targets):
    d = work / f"scene_{name}"
    logf = d / "log.json"
    if not logf.exists():
        return None
    log = json.load(open(logf, encoding="utf-8"))
    if not log:
        return None
    mock = any("(mock)" in r["reason"] for r in log)
    stops = [r for r in log if r["action"] == "stop"]
    ok = bool(stops) and "VLM" not in stops[-1]["reason"]
    dist = None
    if ok:
        bx, by = BOX[targets[-1]]
        dist = math.hypot(stops[-1]["x"] - bx, stops[-1]["y"] - by)
    frames = sorted(d.glob("*.png"))
    return {"n": len(log), "mock": mock, "ok": ok, "dist": dist, "frames": frames,
            "seconds": len(frames) / 50 if frames else None, "log": log}


def thumb(src, dst):
    im = Image.open(src).convert("RGB")
    im.thumbnail((520, 300))
    im.save(dst, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=".")
    ap.add_argument("--pdf", default="説明資料_PAI最終課題.pdf")
    ap.add_argument("--edge", default=None)
    ap.add_argument("--model-name", default="gemini-3.6-flash")
    args = ap.parse_args()
    work = pathlib.Path(args.work_dir)
    out = HERE / "out"
    img_dir = out / "img"
    shutil.rmtree(out, ignore_errors=True)
    img_dir.mkdir(parents=True)

    rows, thumbs, warn = [], [], []
    total_calls, total_sec = 0, 0.0
    for name, title, targets, situ in SCENES:
        s = summarize(work, name, targets)
        if s is None:
            rows.append(f"<tr><td>{name}</td><td>{html.escape(title)}</td><td colspan='3' class='na'>未実施</td></tr>")
            warn.append(f"シーン {name} は未実施です")
            continue
        if s["mock"]:
            warn.append(f"シーン {name} はモック(Gemini なし)の結果です")
        if not s["ok"]:
            warn.append(f"シーン {name} は途中で止まっています")
        total_calls += s["n"]
        total_sec += s["seconds"] or 0
        dist = f"{s['dist']:.1f} m 手前" if s["dist"] is not None else "—"
        sec = f"{s['seconds']:.1f} 秒" if s["seconds"] else "—"
        rows.append(f"<tr><td>{name}</td><td>{html.escape(title)}<br><span class='sub'>{html.escape(situ)}</span></td>"
                    f"<td>{s['n']} 回</td><td>{JA[targets[-1]]}い箱の {dist}</td><td>{sec}</td></tr>")
        if s["frames"]:
            # 止まった瞬間(最後の少し前)の画像を使う
            f = s["frames"][max(0, len(s["frames"]) - 62)]
            dst = img_dir / f"{name}.jpg"
            thumb(f, dst)
            thumbs.append((name, title, f"img/{name}.jpg"))

    # Gemini の判断例(シーンAの最初の数回)
    ex = ""
    a = summarize(work, "A", ["red"])
    if a:
        trs = "".join(
            f"<tr><td>{r['step'] / 50:.0f} s</td><td>{r['action']}</td><td>{html.escape(r['reason'])}</td></tr>"
            for r in a["log"][:5] + a["log"][-1:])
        ex = f"<table class='ex'><tr><th>時刻</th><th>行動</th><th>Gemini が書いた理由(シーンA)</th></tr>{trs}</table>"

    thumbs_html = "".join(
        f"<figure><img src='{p}'><figcaption>{n}: {html.escape(t)}</figcaption></figure>" for n, t, p in thumbs)

    tpl = (HERE / "report_template.html").read_text(encoding="utf-8")
    page = (tpl.replace("{{ROWS}}", "\n".join(rows)).replace("{{THUMBS}}", thumbs_html)
            .replace("{{EXAMPLE}}", ex).replace("{{MODEL}}", html.escape(args.model_name))
            .replace("{{TOTAL_CALLS}}", str(total_calls)).replace("{{TOTAL_SEC}}", f"{total_sec:.0f}"))
    html_path = out / "report.html"
    html_path.write_text(page, encoding="utf-8")

    pdf = out / args.pdf
    subprocess.run([args.edge or edge_path(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
    print("PDF:", pdf)
    for w in warn:
        print("注意:", w)


if __name__ == "__main__":
    main()
