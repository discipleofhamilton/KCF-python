# 追蹤基準測試

以 VOT2016 序列（附 ground truth）離線評估準確度與速度。pysot（SiamRPN）範例影片 `bag.avi`
就是 VOT2016 的 `bag`（垃圾袋）序列，尺寸相同，因此直接沿用其標註；SiamMask 範例的 `tennis`
影格沒有標註，只用於速度與目視檢查。資料不納入版本控制。

```sh
uv run --locked python benchmarks/fetch_vot.py --out benchmarks/data      # 約 190 MB
uv run --locked python benchmarks/evaluate.py --data benchmarks/data/vot \
    --video bag=benchmarks/data/pysot/bag.avi --configs hog_ms_color hog_ms_color_rec
```

- **OPE**：只在第一幀初始化。追蹤器回傳空框即視為之後全部遺失（與 CLI 行為一致），報告平均 IoU、
  success AUC 與 20 px 中心誤差精度（prec20）。
- **Reset**（VOT 監督式）：IoU 為 0 記一次失敗，5 幀後以 ground truth 重新初始化；accuracy 不含
  重新初始化後 10 幀。
- FPS 只計 `init`／`update` 時間，不含解碼；Numba 先暖機。

序列分兩組：`DEVELOPMENT`（16 部，含 bag）用於開發與調參；`HOLDOUT`（12 部）只用來驗證最終選擇。

## 結果（28 部序列，Apple M2 Pro，Python 3.11）

| 設定 | 平均 IoU | AUC | prec20 | Reset 失敗 | FPS |
| --- | --- | --- | --- | --- | --- |
| 原始 `hog_ms_color`（d4f77a2） | 0.339 | 0.337 | 0.454 | 54 | 236 |
| **本版 `hog_ms_color_rec`（建議）** | **0.389** | **0.387** | **0.542** | **49** | **401** |
| 本版 `hog_sf_color`（scale filter） | 0.346 | 0.344 | 0.445 | 58 | 366 |
| 本版 `hog_sf_color_rec` | 0.399 | 0.398 | 0.542 | 49 | 440 |
| 原始 `hog` → 本版 `hog`（追蹤結果相同） | 0.163 | 0.161 | 0.245 | — | 496 → 838 |

FPS 為原始與本版程式交替執行兩次的平均；OPE 指標兩版一致到小數第三位。Reset 失敗數以本版程式、
相同選項量測（方案 A 只改變浮點捨入，失敗數可能差 ±1）。`hog_sf_color_rec` 在 dev 組較佳（0.495／16），held-out 組平均 IoU 相當但失敗較多（0.272／33 對 0.274／28），
因此建議仍採 3 尺度搜尋；需要更高速度時可改用 `--scale-filter`（約快 10%）。scale filter 在信心不足的幀不估計尺度，
以免遮擋時框往遮擋物縮放（此修正前為 0.375／56）。分組來看，相較同版未啟用 recovery 的
`hog_ms_color`：dev 組 0.445／17 次失敗 → 0.475／21；held-out 組 0.198／37 → 0.274／28。增益集中在少數序列（blanket、shaking、
pedestrian1：原本因峰值低於門檻而永久遺失），28 部序列仍屬小樣本，差距 ±2 次失敗屬雜訊範圍。

## 方案評估過程

以五個子代理各自實作並量測一個方案，再交叉辯論：

| 方案 | 內容 | 結論 |
| --- | --- | --- |
| A 頻域批次 FFT | 31 個 HOG 通道一次 `rfft2`、頻域加總後單次反轉換、快取範本頻譜；fHOG 微調且輸出位元相同 | **採用**：追蹤結果不變，約 1.7× |
| B fDSST 尺度濾波器 | 單次平移偵測＋1-D 尺度相關濾波（17→33 尺度） | **選用**（`--scale-filter`）：dev 組較佳，held-out 組變差 |
| C 外觀特徵 | Color Names 特徵、HSV／Lab 直方圖、適應性融合 | **不採用**：CN 變慢且無增益；色彩空間增益依賴單一序列與融合比例 |
| D 信心門控與重新偵測 | APCE／峰值低時不更新模型；低於門檻改為擴大範圍重新偵測，不再永久放棄 | **採用**（`--recovery`）：唯一在 held-out 組也穩定改善的方案 |
| E OpenCV DNN（NanoTrack 等） | `cv2.TrackerNano`／`TrackerVit`／`TrackerDaSiamRPN` | **不納入**：NanoTrack 準確度最高（dev 組 0.600），但權重未聲明授權且非 KCF；僅作參考 |

辯論後否決的細節：常速度運動先驗（bag 由 0.55 降到 0.27）、低信心幀也做重新偵測（失敗增加）、
低信心幀降速學習（28 部合計無明顯增益）、融合比例 0.4（只在 dev 組有利）。

## 平台驗證

- macOS arm64：Python 3.11。
- Linux aarch64、x86_64（`docker/run-tests.sh`，x86_64 為模擬）：Python 3.11／3.12，`uv sync --locked`、Ruff、pytest
  全數通過，基準測試可執行。不同 CPU 架構的浮點差異會使個別序列結果略有不同。
- Windows x86_64：未實測；`uv.lock` 中 NumPy、Numba、llvmlite、OpenCV 皆有 `win_amd64` wheel。
  Windows on ARM 缺少 Numba 與 OpenCV wheel（既有限制）。

## 資料來源

- VOT2016：<https://www.votchallenge.net/vot2016/>（序列由 VOT 依原始來源授權提供，僅供研究評估）。
- pysot `demo/bag.avi`：<https://github.com/STVIR/pysot>（Apache-2.0）。
- SiamMask `data/tennis`：<https://github.com/foolwood/SiamMask>（MIT）。
