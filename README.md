# 基于文本的视频异常行为检测

本项目按 doc/基于文本的视频异常行为检测_评估与修订实施规划.docx 推进。主任务是给定视频和事件描述，输出相应时间段与连续分数。

## 当前状态

- 本地设备：RTX 4060 Laptop GPU，8 GB 显存。
- 本地 Python：.conda/local，Python 3.10.21。
- 官方参考仓库：third_party/Paper-AnyAnomaly，提交 ec7e9fc36e56ca16a5fa08386e368a7e4d18dbb0。
- 正式 Qwen/vLLM 推理计划在云端 Linux GPU 上进行。

## 下一步

1. 获取 CUHK Avenue 原始数据，核对官方仓库 ground_truth 中 C-Ave 的五类标注。
2. 检查视频名称排序、帧数和标签长度。
3. 在本地完成短视频解码与 CLIP 小样本测试。

大型数据、模型权重、运行结果和本地环境不提交 Git。
