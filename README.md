# 基于文本的视频异常行为检测

本项目按 doc/基于文本的视频异常行为检测_评估与修订实施规划.docx 推进。主任务是给定视频和事件描述，输出相应时间段与连续分数。

## 当前状态

- 本地设备：RTX 4060 Laptop GPU，8 GB 显存。
- 本地 Python：.conda/local，Python 3.10.21；本地评估所需的 matplotlib、SciPy、scikit-learn、fastprogress 已安装并核验。
- 官方参考仓库：third_party/Paper-AnyAnomaly，提交 ec7e9fc36e56ca16a5fa08386e368a7e4d18dbb0。
- 正式 Qwen/vLLM 推理计划在云端 Linux GPU 上进行。
- Avenue 测试视频已转为 15,324 张 JPG；常规五类事件对应 18 个视频，按 24 帧分段共有 507 个完整片段。

## 下一步

1. 整理可重复运行的单视频 CLIP 检查，明确文本、输出和尾帧处理。
2. 正式 Qwen/vLLM 评分前，确认 Linux GPU 环境和软件、模型下载位置。
3. 正式评估需覆盖规定的视频与事件，并说明不足 24 帧的尾段如何处理。

大型数据、模型权重、运行结果和本地环境不提交 Git。Windows 本地测试的依赖版本见 [requirements-windows.txt](requirements-windows.txt)。

## 工作记录

每次工作的实际进展和下一步都写在 [工作记录.md](工作记录.md)，按日期保留历史。

## GitHub 同步

公开仓库：[text-guided-video-anomaly-detection](https://github.com/c37958640-hash/text-guided-video-anomaly-detection)。每次工作结束更新工作记录，按需提交代码和文档，再推送 main 到 GitHub。
