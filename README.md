# 基于文本的视频异常行为检测

本项目按 doc/基于文本的视频异常行为检测_评估与修订实施规划.docx 推进。主任务是给定视频和事件描述，输出相应时间段与连续分数。

## 当前状态

- 本地设备：RTX 4060 Laptop GPU，8 GB 显存。
- 本地 Python：.conda/local，Python 3.10.21；本地评估所需的 matplotlib、SciPy、scikit-learn、fastprogress 已安装并核验。
- 官方参考仓库：third_party/Paper-AnyAnomaly，提交 ec7e9fc36e56ca16a5fa08386e368a7e4d18dbb0。
- 正式 Qwen/vLLM 推理计划在云端 Linux GPU 上进行。
- Avenue 测试视频已转为 15,324 张 JPG；常规五类事件对应 18 个视频，按 24 帧分段共有 507 个完整片段。

## 下一步

1. 检查多视频结果中的低分案例，先看跳舞 18 号和靠近镜头 19 号的片段、标签与分数变化。
2. 根据失败案例决定下一轮改进；正式 Qwen/vLLM 评分前，确认 Linux GPU 环境和软件、模型下载位置。
3. 正式评估继续写清不同视频、事件及不足 24 帧尾段的处理口径。

## 单视频 CLIP 检查

在项目根目录运行下面的命令；不加参数时使用第 16 号视频、骑车标签和文字 `a person riding a bicycle`：

```powershell
.\.conda\local\python.exe .\scripts\clip_single_video_diagnostic.py
```

如需更换视频或事件文字，可使用 `--video-id` 和 `--text`。当文字描述的是其他事件时，也要用 `--event` 选择对应标签，可选 `bicycle`、`dancing`、`running`、`throwing`、`too_close`。运行 `--help` 可查看全部参数。脚本使用项目中已有的 JPG 帧和 CLIP 权重，不会自行下载模型。

结果保存在 `outputs/clip_16_bicycle.json`（参数改变时文件名也相应改变），列出每个完整 24 帧片段的起止帧和分数；末尾不足 24 帧的部分不参与 AUROC，并单独统计被排除的正例。若纳入评估的帧只有一种标签，AUROC 显示为无法计算，片段分数仍会保存。这是 CLIP 单视频诊断，不是 AnyAnomaly 的 Qwen/vLLM 正式结果。

## 五类事件的多视频检查

五句固定事件描述在 [配置文件](configs/avenue_clip_prompts.json)，多视频脚本会把 18 个常规视频分别与五句描述比较。先在项目根目录运行：

```powershell
.\.conda\local\python.exe -m scripts.evaluate_clip_avenue
```

结果说明、五类整体 AUROC 和含正例视频的单视频结果见 [Avenue CLIP 检查报告](reports/avenue_clip_5events_2026-09-25.md)。详细片段分数保存在本地 `outputs/clip_avenue_5events.json`，不会提交到 Git。这仍是 CLIP 探索性检查，不是 AnyAnomaly 论文的完整基线。

大型数据、模型权重、运行结果和本地环境不提交 Git。Windows 本地测试的依赖版本见 [requirements-windows.txt](requirements-windows.txt)。

## 工作记录

每次工作的实际进展和下一步都写在 [工作记录.md](工作记录.md)，按日期保留历史。

## GitHub 同步

公开仓库：[text-guided-video-anomaly-detection](https://github.com/c37958640-hash/text-guided-video-anomaly-detection)。每次工作结束更新工作记录，按需提交代码和文档，再推送 main 到 GitHub。
