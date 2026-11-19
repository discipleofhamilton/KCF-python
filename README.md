# KCF tracker in Python 3.11 / 3.12

以 Python 實作 KCF（Kernelized Correlation Filters）單一物件追蹤。
開啟攝影機或影片後，以滑鼠指定目標，逐幀更新追蹤框並顯示 FPS。
支援灰階／HOG 特徵、多尺度估計，以及實驗性的背景去除。

本專案源自 [uoip/KCFpy](https://github.com/uoip/KCFpy)，其實作來自
[KCFcpp](https://github.com/joaofaro/KCFcpp)，演算法參考
[High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)。
授權見 [LICENSE](LICENSE)。

## 環境與安裝

依照[開發工具使用手冊](https://ionetworks.slite.page/p/p5chrq_Z79)導入 uv、Ruff、pre-commit 與 pytest。
支援範圍為 Python `>=3.11,<3.13`；預設請求 3.11，已在 3.11、3.12 執行測試。
尚未宣告支援 3.13 以上；擴充範圍時應同步更新 lock 並驗證各支援版本。

先依 [uv 官方指南](https://docs.astral.sh/uv/getting-started/installation/)安裝 uv，在專案根目錄執行：

```sh
uv sync --locked
uv run --locked python --version
uv run --locked kcf-track --help
uv run --locked pre-commit install
```

執行依賴只有 NumPy、Numba 與 OpenCV；使用具有視窗支援的 `opencv-python`。
不要在同一環境混裝其他 OpenCV 發行套件。Linux 執行環境可能另需 `libgl1`、`libglib2.0-0` 及可用的圖形介面。
攝影機亦需作業系統授權。第一次使用 HOG／背景去除時，Numba 編譯可能造成短暫停頓。

`requirements.txt` 僅保留 `-e .` 相容入口；依賴宣告仍以 `pyproject.toml` 為準。
團隊重建環境應使用 uv lock 流程，`pip install -r requirements.txt` 不會重現 lock 的精確版本。

## 使用

```sh
# 預設攝影機
uv run --locked kcf-track

# 指定攝影機，啟用 HOG 與多尺度
uv run --locked kcf-track --source 2 --use-hog --multiscale

# 影片與背景去除
uv run --locked kcf-track --source ./test.avi --use-hog --use-debackground

# 仍可直接執行既有入口
uv run --locked python run.py --source ./test.avi
```

操作：第一幀暫停，左鍵拖曳框選目標後，按任意鍵開始播放；`q`／`Esc` 直接離開。
播放中可重新左鍵框選，或用右鍵將上次框選的大小移到指定中心位置。
框選寬、高都必須大於 10 像素。追蹤信心不足時會停止畫框，需重新指定目標。

| 選項 | 功能 |
| --- | --- |
| `--source` | 攝影機索引或影片路徑，預設 `0` |
| `--use-hog` | 使用 HOG；預設使用灰階特徵 |
| `--fixed-window` | 將範本的最長邊固定為約 96 像素，尺寸仍會對齊特徵網格 |
| `--multiscale` | 比較原大小及相鄰兩個尺度；啟用時採固定範本尺寸 |
| `--use-debackground` | 擷取特徵前套用背景遮罩，預設關閉 |

不支援自動物件分類、圖片資料夾序列載入或同時追蹤多個目標。

背景去除與前景擷取也能獨立操作：

```sh
uv run --locked kcf-debackground --source 0 --size 40 30
uv run --locked kcf-debackground --use-custombitwise
uv run --locked python foreground_filter.py ./input_image.jpg
```

背景去除將影像切成「列 × 欄」網格、量化平均色彩，從邊界以四向連通擴展相同顏色的背景。
網格不超過影像尺寸；遮罩放大後套回原圖，保留前景紋理、`uint8` 型別與 0–255 範圍。
邊界屬於背景是演算法假設，目標碰到邊界或與背景顏色相近時可能被誤刪。

## 程式結構

保留原有根目錄模組結構，透過 setuptools 的 `py-modules` 安裝；`src/` 不是目前的安裝入口。

| 檔案 | 責任 |
| --- | --- |
| `run.py` | CLI、滑鼠選取狀態與影像迴圈 |
| `kcftracker.py` | `KCFTracker.init(roi, image)`、`update(image)` 及模型更新 |
| `fhog.py` | Numba 加速的 HOG 特徵計算 |
| `fft.py` | OpenCV DFT 與複數運算 |
| `recttool.py` | 區域裁切、邊界填補及背景遮罩整合 |
| `debackground.py` | 色彩網格、背景連通與比較展示 |
| `foreground_filter.py` | 獨立的 Otsu／最大輪廓前景擷取 |
| `utils.py` | 遮罩運算、距離與計時工具 |
| `tests/` | 合成影像追蹤、特徵運算與 CLI 回歸測試 |

`roi` 格式為 `[x, y, width, height]`。
`update()` 信心不足時回傳 `[0, 0, 0, 0]`，保留最後模型，呼叫端應停止追蹤並等待重新 `init()`。
內部函式及輔助方法已改為 snake_case；若外部程式曾直接使用原本的 camelCase 名稱，需同步更新。

## 開發與驗證

```sh
uv sync --locked
uv run --locked ruff check --fix .
uv run --locked ruff format .
git diff

# 提交前的品質檢查；不修改原始碼
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pre-commit validate-config
uv run --locked pytest
```

Ruff 採 Python 3.11 語法目標、120 字元排版目標與 `E,F,I,UP,B,N,ASYNC,LOG` 規則；忽略 `E501`。
因實際程式在根目錄，檢查範圍使用 `.`，涵蓋根目錄模組與測試。
pre-commit 以 system hook 呼叫 lock 中的 Ruff；每個 checkout 仍需自行安裝 hook，且 Git／IDE 的 PATH 必須找得到 uv。
hook 改檔後先檢視差異，再重新 stage 與提交；不要用 `--no-verify` 略過檢查。
測試使用合成影像與替代 GUI 呼叫，不會開啟攝影機或要求顯示器。

新增／移除依賴使用 `uv add`／`uv remove`；更新後一併檢視 `pyproject.toml` 與 `uv.lock`。
不要手動編輯 lock，或以 `--frozen` 隱藏宣告與 lock 不一致的問題。
詳細檔案分工與 review 流程見 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 限制

KCF 適合短期追蹤，沒有完整的遮擋恢復或自動重新偵測機制。
合成影像測試通過不等於真實影片的精度或 FPS 已達特定標準；背景去除亦需依場景評估。
