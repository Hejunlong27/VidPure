# Benchmark

> 欢迎提交实测数据 PR：跑 `python scripts/benchmark.py --input your.mp4`，
> 把输出表格贴进 PR 即可。以下为预估基线，随版本更新。

测试对象：1080p · 30fps · 30s · 900 帧；输出 CRF18 H.264；音频直通。

| 平台 | clean 通道 | upscale | clean 耗时 | upscale 耗时 | 合计 | 备注 |
|---|---|---|---|---|---|---|
| RTX 40/50 · 8GB · Win11 | LaMa (cuda) | animevideov3 x2 tile256 (NCNN) | _预估 5~8 min_ | _预估 3~6 min_ | _12~20 min_ | 默认档 |
| RTX 40/50 · 8GB | delogo | 同上 | 2~3 min | 3~6 min | 5~10 min | 快速通道 |
| 核显 / CPU | delogo | NCNN tile192 | 2~3 min | 8~15 min | 10~18 min | 无独显 |
| RTX 16GB+ | ProPainter | NCNN tile384 | 10~20 min | 2~5 min | 15~25 min | 视频级修复 |

## 复现

```bash
python scripts/benchmark.py --input sample.mp4 --clean lama --scale 2
```

脚本将依次单独计时 clean 与 upscale 两阶段，输出 Markdown 行。
