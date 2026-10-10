# ローカル GPU（Windows + RTX 50 系）で追加実験を回す手順

友人の PC（Windows + RTX 5070 Ti など）で、`scripts/run_experiments.py` により追加実験 9 本を一晩で回す手順。
結果は `results/` を zip にして返してもらう。

**注意: この手順は Windows の実機では確認していない。** Python 側のコマンド組み立てはテスト
（`test_run_experiments.py`）と dry-run で確認したが、Windows 上での実行・学習は未確認。
うまくいかないときは末尾の「困ったとき」を見る。

## 回す実験（`scripts/experiments_local.json`）
| 出力名 | 内容 |
|---|---|
| `yolox_s_windfarm_s2`, `_s3` | ベースライン、seed 2, 3 |
| `yolox_s_windfarm_ecbam_s2`, `_s3` | 施策 D（E-CBAM）、seed 2, 3 |
| `yolox_s_windfarm_portrait_s2`, `_s3` | 施策 E（縦長写真の RFS）、seed 2, 3 |
| `yolox_s_windfarm_ecbam_portrait_s1` | D+E の組み合わせ、seed 1 |
| `yolox_s_windfarm_ecbam_in800_s1` | D を入力 800px（`input_size` / `test_size` = 800x800）、seed 1 |
| `yolox_s_windfarm_ecbam_e100_s1` | D を 100 エポック（`max_epoch` = 100）、seed 1 |

seed を指定した学習は cuDNN の deterministic モードが有効になり、遅くなることがある（YOLOX の仕様）。

## 1. PyTorch（RTX 50 系は CUDA 12.8 以上が必要）
RTX 50 系（Blackwell、sm_120）は CUDA 12.8 以上に対応した PyTorch（2.7 以降、cu128 ホイール）が要る。
古い PyTorch だと `no kernel image is available for execution on the device` のようなエラーになる。
Python 3.10〜3.12 の仮想環境を作り、先に PyTorch を入れる（NVIDIA ドライバは最新にしておく）:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`True` と GPU 名が出ることを確認する。

## 2. YOLOX と依存
```powershell
git clone https://github.com/Megvii-BaseDetection/YOLOX
git clone https://github.com/DanGaRa215/YOLOX yolox_task    # このリポジトリ（名前は任意）
cd YOLOX
```
`requirements.txt` から `onnx` を含む行を消す（onnx / onnx-simplifier は新しい Python でビルドに失敗しやすく、今回は使わない）。
エディタで消すか、PowerShell なら:
```powershell
(Get-Content requirements.txt) | Where-Object { $_ -notmatch 'onnx' } | Set-Content requirements.txt
pip install -r requirements.txt
pip install pycocotools thop ninja loguru tabulate tensorboard
```
Windows で `pycocotools` のビルドに失敗する場合は `pip install pycocotools-windows` も試す。
`pip install -e .` が通ればそれでよいが、通らなければインストールせず `PYTHONPATH` で読み込む
（`run_experiments.py` は `--yolox-dir` を子プロセスの `PYTHONPATH` に自動で入れるので、通常は何もしなくてよい）。

事前学習済みの重みを YOLOX のフォルダに置く:
```powershell
curl.exe -L -o yolox_s.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

## 3. 環境変数
PyTorch 2.6 以降で YOLOX のチェックポイントを読み込むために必要（`run_experiments.py` が子プロセスに自動で設定するが、
手でコマンドを打つときのために）:
```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
```

## 4. データの準備
Roboflow の "Wind Farms" v5（YOLO 形式）を展開し、COCO 形式に変換する。symlink 版は Windows で権限の問題が出るので `--mode copy` を使う:
```powershell
cd ..\yolox_task
python convert_yolo_to_coco.py --src <Roboflow の展開先> --out datasets\windfarm --mode copy
python verify_coco.py    # 任意: 件数の確認
```
`datasets\windfarm` に `train2017 val2017 test2017 annotations` ができる（train 2643 / valid 247 / test 130 枚）。
すでに変換済みの zip をもらっている場合は、`datasets\windfarm` に展開するだけでよい。

## 5. 実行
このリポジトリのフォルダで実行する。まず dry-run でコマンドを確認する:
```powershell
python scripts\run_experiments.py --yolox-dir ..\YOLOX --data-dir datasets\windfarm --out-dir outputs_local --dry-run
```
問題なければ本番（9 本を順に学習・評価）:
```powershell
python scripts\run_experiments.py --yolox-dir ..\YOLOX --data-dir datasets\windfarm --out-dir outputs_local
```
- 引数は環境変数でも指定できる: `YOLOX_DIR`, `YOLOX_DATA_DIR`, `OUT_DIR`, `PRETRAINED`（既定は `<yolox-dir>\yolox_s.pth`）。
- 先に 1 本だけ試すなら `--only yolox_s_windfarm_s2`。
- `--occupy` を付けると学習に `-o`（GPU メモリの先取り）が付く。既定はオフ。
- ログは画面と `outputs_local\run_experiments.log`（全体）、`outputs_local\<name>\train_stdout.log`（学習）に出る。
- 1 本ごとに `outputs_local\<name>\`（`best_ckpt.pth` など）と `outputs_local\results\`（`<name>_{val,test}.json`、`_preds.json`、`_cocomap.txt`）ができる。

### 中断したとき
同じコマンドをもう一度実行すればよい。
- `outputs_local\<name>\DONE` がある実験は飛ばす。
- `latest_ckpt.pth` があれば `--resume` で続きから学習する。
- 学習が終わっていて評価だけ残っている実験は、評価だけやり直す。
- 失敗した実験があっても次の実験へ進み、最後に失敗した名前を表示する（終了コードは 1）。

## 6. 結果の返し方
`outputs_local\results\` を zip にして送る（ckpt は大きいので不要）:
```powershell
Compress-Archive -Path outputs_local\results -DestinationPath results_local.zip
```
受け取った側は、既存の `results/` に展開して `summarize_results.py` で集計する。

## 困ったとき（実機未確認のため、見る点）
- **CUDA エラー / `no kernel image`**: PyTorch が cu128 版か確認（手順 1）。ドライバを更新する。
- **dataloader が固まる・`BrokenPipe` / `spawn` 系のエラー**: Windows ではワーカープロセスの起動が重い。
  exp の `data_num_workers`（YOLOX 既定 4）を減らす。例: 実験リストの `opts` に `{"data_num_workers": 2}` や `0` を足す。
- **メモリ不足（`--cache ram`）**: 縮小済みの画像を RAM に載せる（約 1.7GB。入力 800px では増える）。足りなければ
  `scripts\run_experiments.py` の `--cache ram` を外すか、`-b 8` にする（コードの `build_train_cmd` を編集）。
- **GPU メモリ不足（OOM）**: `-b 16` を `-b 8` に下げる。`--occupy` を付けていないか確認する。
- **パス区切り**: スクリプトは `pathlib` で組み立てるが、`--yolox-dir` などに渡すパスの空白は引用符で囲む。
  日本語や空白を含むフォルダ名は避けるのが無難。
- **学習がほとんど進まない / 遅い**: seed を指定すると cuDNN が deterministic になり遅くなる。仕様。
