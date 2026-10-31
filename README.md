# VLM の指示で荷物まで歩く四足ロボット (Physical AI 応用編1 最終課題)

日本語の指示(「赤い荷物のところまで歩いて、近くまで来たら止まって」)を、VLM(Gemini)が
**カメラ画像だけを見て**「前進 / 左 / 右 / 停止」に変換し、強化学習で学習した歩行方策(Unitree Go2)が実行します。
物理シミュレータは [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) を使っています。

```
日本語の指示 + カメラ画像 ──▶ Gemini (VLM) ──▶ forward / left / right / stop
                                                      │ (0.0〜0.8 m/s の前進、-1〜1 rad/s の旋回の指令)
                                                      ▼
                                        歩行方策 (PPO, Colab の GPU で学習) ──▶ Go2 の関節
```

## フォルダ構成

| パス | 内容 |
|---|---|
| `colab/step1_go2_walk.ipynb` | Colab(GPU)で歩行方策を学習するノートブック(数分〜十数分) |
| `src/vlm_walk.py` | VLM の呼び出し、シーンへの箱の追加、動画用オーバーレイなど(共通部分) |
| `src/vlm_scenes.py` | シーンの定義、実行、動画へのつなぎ込み |
| `logs/go2-walking/` | 学習済みの歩行方策 (`model_300.pt`) と設定 (`cfgs.pkl`) |
| `results/scene_logs/` | 実行時の Gemini の判断ログ(時刻、位置、行動、理由) |
| `report/` | 説明資料(PDF)を作るスクリプト |

## 再現手順(Windows / Python 3.13 で確認、GPU なし)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Genesis の公式サンプルから、環境ファイルを取ってくる(Apache-2.0)
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
copy Genesis\examples\locomotion\go2_env.py .
```

`go2_env.py` は、そのまま置けば十分です。実行時に、`src/vlm_walk.py` が、動画用のカメラと2つの箱を追加するように自動で書き換えます。
(動作確認したのは Genesis 1.4.4 と、2026-10-20 時点の公式リポジトリの `go2_env.py` です。)

### 1. Gemini なしの動作確認(モック)

```powershell
python src/vlm_scenes.py --mock --scenes A
```

位置の座標から計算した判断を使います。**AI は使っていません。** 動作確認用です。

### 2. Gemini で実行

API キーは、環境変数で渡します(コードには書きません)。

```powershell
$env:GEMINI_API_KEY = "<自分のキー>"
python src/vlm_scenes.py --scenes A --model gemini-3.6-flash
python src/vlm_scenes.py --scenes B --model gemini-3.6-flash
python src/vlm_scenes.py --scenes C1 --model gemini-3.6-flash
python src/vlm_scenes.py --scenes C2 --model gemini-3.6-flash   # C1 が止まった位置から続ける
python src/vlm_scenes.py --scenes D --model gemini-3.6-flash
python src/vlm_scenes.py --assemble                              # final.mp4 にまとめる
```

- 無料枠は、モデルごとに「1日20回」「1分5回」の上限があるので、シーンごとに分けて実行できるようにしてあります。
  呼び出しの間隔は、既定で13秒です(`--min-interval`)。
- 失敗したシーン(VLM が使えず途中で止まった)は、`scene_X_failed/` に退避され、動画には混ざりません。

### 3. 歩行方策を学習し直す場合

`colab/step1_go2_walk.ipynb` を Google Colab で開き(ランタイムは GPU)、上から順に実行します。
最後のセルで、`go2_export.zip` がダウンロードされます。中の `logs/` を、このリポジトリの `logs/` に置き換えます。

> 注意: Colab の GPU 環境では、Genesis の描画(EGL)の初期化が落ちるため、
> ノートブックでは、学習のときだけ描画の準備を無効にしています。動画は、手元の PC で作ります。

## シーン

| シーン | 内容 |
|---|---|
| A | 赤い箱が見えている状態から、赤い荷物へ |
| B | 同じ場所から、指示だけを変えて、青い荷物へ |
| C1, C2 | 荷物が見えない後ろ向きから赤い荷物を探して止まる(C1)。そこから、指示を青に切り替えて、青い荷物へ(C2) |
| D | 目の前の赤い箱を無視して、後ろにある青い箱を探す |

## 既知の制限

- VLM に渡す画像は、ロボットの後ろ上方からの視点(ロボットの体も映る)です。ロボット自身に載ったカメラの映像ではありません。
- 荷物(箱)の位置はシミュレータ内で固定です。判断は画像だけで行い、座標は VLM に渡していません。
- 歩行方策は、300回の学習(短め)です。速い旋回と前進を同時に出すと、不安定になります(指令を控えめにしています)。
- 結果は、ここに載せた実行1回分です。VLM の判断は、実行ごとに少し変わります。

## ライセンス・出典

- 環境ファイル `go2_env.py`、Go2 のモデルは、Genesis (Apache-2.0) のものです。
- 本リポジトリの `src/` は、課題提出用に作成したものです。
