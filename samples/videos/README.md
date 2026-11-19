# 測試影片

三個本機片段各 20 秒，皆為 H.264 MP4、24 FPS、無音訊。
檔案從網路上的完整影片擷取，縮小至 640 像素寬，並非原作者發行的完整作品。

| 檔案 | 原片範圍 | 解析度 | 大小 | 測試情境 |
| --- | --- | --- | --- | --- |
| [big_buck_bunny_20s.mp4](big_buck_bunny_20s.mp4) | 00:50–01:10 | 640 × 360 | 1,048,326 bytes | 兔子移動、姿態及尺度變化 |
| [sintel_20s.mp4](sintel_20s.mp4) | 03:20–03:40 | 640 × 366 | 1,429,302 bytes | 角色與龍、背景紋理及鏡頭切換 |
| [tears_of_steel_20s.mp4](tears_of_steel_20s.mp4) | 00:40–01:00 | 640 × 366 | 1,735,087 bytes | 真人、複雜背景及場景切換 |

影片含鏡頭切換，追蹤失敗時需重新框選；它們是手動操作樣本，沒有 ground-truth 標註，不適合直接計算追蹤準確率。
已以 OpenCV 逐幀解碼確認每部皆為 480 幀、20 秒。

在專案根目錄執行：

```sh
uv run --locked kcf-track --source samples/videos/big_buck_bunny_20s.mp4 --use-hog --multiscale
uv run --locked kcf-track --source samples/videos/sintel_20s.mp4 --use-hog
uv run --locked kcf-track --source samples/videos/tears_of_steel_20s.mp4 --use-hog
```

第一幀以左鍵框選目標後按任意鍵播放；播放中可以重新框選。

## 來源與授權

這些媒體使用原作品的 **Creative Commons Attribution 3.0（CC BY 3.0）** 授權，
不適用本專案程式碼的 MIT 授權。分享片段時應保留以下署名、來源、授權連結及修改說明。

- **Big Buck Bunny** — © copyright 2008, Blender Foundation / www.bigbuckbunny.org。
  [原作及授權說明](https://peach.blender.org/about/)、[官方下載頁](https://peach.blender.org/download/)。
- **Sintel** — © copyright Blender Foundation | www.sintel.org。
  [原作及授權說明](https://durian.blender.org/sharing/)、[官方下載頁](https://durian.blender.org/download/)。
- **Tears of Steel** — © Blender Foundation | mango.blender.org。
  [原作及授權說明](https://mango.blender.org/about/)、[官方下載頁](https://mango.blender.org/production/tears-of-steel-download-watch/)。

共同授權：[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)。
本專案的修改：依上表截取 20 秒、縮圖、轉為 24 FPS H.264／yuv420p、移除音訊。
這些測試用途及修改不代表原作者對本專案的背書。

下載日期：2026-11-19。原先的 Google 公開範例下載連結回傳 403，實際採用
[DemoVideos 公開鏡像](https://github.com/andreasbotsikas/DemoVideos)：

- [Big Buck Bunny 原始檔](https://raw.githubusercontent.com/andreasbotsikas/DemoVideos/master/big_buck_bunny.mp4)
- [Sintel 原始檔](https://raw.githubusercontent.com/andreasbotsikas/DemoVideos/master/sintel.mp4)
- [Tears of Steel 原始檔](https://raw.githubusercontent.com/andreasbotsikas/DemoVideos/master/tears_of_steel.mp4)

本機片段 SHA-256：

```text
7d483a983a2d796825c2b2ced1033da2adba358c98c17bfb272647659dd2f7b2  big_buck_bunny_20s.mp4
1e6194b42c8006f21a0f35c761a7e49b69ac044690f8ba8c8eb1e5fe516fc428  sintel_20s.mp4
2d45ef18be18622095fcc5edf31ef61ade1d90de5c7b15a1a9ad05d70f64cf20  tears_of_steel_20s.mp4
```
