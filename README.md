# 基于文本的视频异常行为检测

本项目按 doc/基于文本的视频异常行为检测_评估与修订实施规划.docx 推进。主任务是给定视频和事件描述，输出相应时间段与连续分数。

## 当前状态

- 本地设备：RTX 4060 Laptop GPU，8 GB 显存。
- 本地 Python：.conda/local，Python 3.10.21。
- 官方参考仓库：third_party/Paper-AnyAnomaly，提交 ec7e9fc36e56ca16a5fa08386e368a7e4d18dbb0。
- 正式 Qwen/vLLM 推理计划在云端 Linux GPU 上进行。

## 下一步

1. 把 Avenue 视频整理为参考代码读取的 JPG 帧目录。
2. 核对帧序、标签、24 帧分段和尾部帧处理。
3. 在少量 JPG 帧上对照 AnyAnomaly 的 CLIP 预处理，再准备完整评估。

大型数据、模型权重、运行结果和本地环境不提交 Git。Windows 本地测试的依赖版本见 [requirements-windows.txt](requirements-windows.txt)。

## 工作记录

每次工作的实际进展和下一步都写在 [工作记录.md](工作记录.md)，按日期保留历史。
