# 開發約定

使用 Python 3.11／3.12 與 uv；初次加入執行 `uv sync --locked`、`uv run --locked pre-commit install`。
本次驗證使用 uv 0.11.14；uv 執行檔本身不由 `uv.lock` 鎖定。

| 檔案 | 維護責任 |
| --- | --- |
| `pyproject.toml` | Python 支援範圍、依賴、安裝入口、Ruff 與 pytest 設定 |
| `uv.lock` | uv 產生的直接／間接依賴解析結果，應隨依賴變更納入版本控制 |
| `.python-version` | 本機預設請求 3.11，不固定 patch |
| `.pre-commit-config.yaml` | 本機提交前的 Ruff 修正與排版 |

保留 KCF、HOG、多尺度與背景去除的既有功能，避免無關的新依賴或抽象。
修正行為時新增能重現問題的小型測試；外部影像與 GUI 互動以合成資料及 monkeypatch 隔離。
核心演算法與滑鼠互動的實測結果應與無顯示器測試分別說明。

送 review 前執行：

```sh
uv sync --locked
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked pre-commit validate-config
uv run --locked pytest
```

Ruff 通過不能取代測試或型別檢查。新增規則時應理解診斷並修正程式，不以廣泛 suppression 或 unsafe fixes 換取綠燈。
本專案沒有另設 mypy、coverage 門檻或硬體效能門檻。

合併前應完成上述檢查，並由 reviewer 核對行為與相容性。

支援版本調整需同步修改 Python 宣告、`.python-version`（若預設變更）、Ruff 目標與 lock，並在各宣告版本實際執行測試。
