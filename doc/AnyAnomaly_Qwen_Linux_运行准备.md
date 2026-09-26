# AnyAnomaly 的 Qwen 正式运行准备（Avenue）

更新日期：2026-09-26。这里是下一阶段的执行说明，主规划仍见 `doc/基于文本的视频异常行为检测_评估与修订实施规划.docx`。目前已经加入 `scripts/anyanomaly_qwen_single_video.py` 并在 Windows 上完成第 16 号视频的帧与标签离线检查；尚未安装 Qwen 依赖、下载 Qwen 模型或运行真正的 Qwen 推理。

## 这一步要做什么

先用论文作者的 AnyAnomaly 代码跑通一段视频，再完成 Avenue 五类事件评估。它会读入一段视频对应的连续图片，用 CLIP 挑选关键画面并制作辅助画面，再让 Qwen 根据指定事件文字给出分数。此前的 CLIP 检查仅用于发现问题，两种成绩分开保存、分开报告。

参考代码保存在 `third_party/Paper-AnyAnomaly`，已核对的版本为 `ec7e9fc36e56ca16a5fa08386e368a7e4d18dbb0`。本地参考仓库不纳入 Git，后续 Linux 端要记录同一提交号。

## 现有材料与必须补齐的材料

| 材料 | 当前状态 | 正式运行时的用途 |
| --- | --- | --- |
| Avenue 测试帧 | Windows 项目内 `datasets/avenue/testing/frames/01` 至 `21`，共 15,324 张 JPG | 作者代码按视频文件夹读取 JPG |
| 五类标签及排除清单 | `third_party/Paper-AnyAnomaly/ground_truth` | 核对异常发生在哪些帧；常规评估排除 06、09、11 号复合事件视频 |
| CLIP ViT-B/32 权重 | `models/clip/ViT-B-32.pt` | 挑选关键帧、制作辅助画面 |
| Qwen2.5-VL-3B-Instruct 权重 | 尚未下载 | 正式 Qwen 打分 |
| Linux GPU 与 Qwen 依赖 | 尚未确定 | 运行 vLLM 和完整方法 |

GitHub 只有代码与说明；上述数据、模型和运行结果在 `.gitignore` 中，不会随着 Git 拉取到 Linux。转移时须单独复制或挂载。vLLM 官方当前要求 Linux；本机 Windows 尚未配置 WSL，8 GB 显存能否完整运行本方法也未实测，因此先选 Linux GPU 环境，再根据实际显存试运行。不要依据“3B”直接断定 8 GB 足够或不足。

## 作者代码实际读取和写入的位置

在 `third_party/Paper-AnyAnomaly` 目录运行时：

- `config.py` 的 `data_root` 目前写死为 `/home/anonymous/datasets`。Linux 端须改成**实际**数据根目录；预期其下为 `avenue/testing/frames/01` 等 21 个视频文件夹。
- `ground_truth/c-avenue-multiple.json` 列出要排除的复合事件视频；`ground_truth/c-avenue_labels/avenue_<事件>_label.h5` 是五类标签。Linux 端的工作目录必须让这些相对路径可用。
- 默认 `--model_path Qwen/Qwen2.5-VL-3B-Instruct` 是远程模型名称，可能触发下载。必须改用经用户确认后放好的**本地模型绝对路径**。
- 作者脚本 `clip.load('ViT-B/32')` 也可能使用默认缓存或下载。运行前须改为指向已转移的 `ViT-B-32.pt` 绝对路径，并确认它能成功加载。
- 默认结果写到参考仓库的 `results/avenue/<事件>/3/qwen_vllm_proposed_avenue_<事件>_3.json`；其中每个视频有 `scores`、`scores_wa`、`scores_tc` 三路逐帧分数。图也写在对应目录。运行后把必要结果及运行参数整理到项目的 `outputs/anyanomaly/`；该目录只留本地，不推送 GitHub。

参考仓库的 `requirements.txt` 只有 h5py、fastprogress、scikit-learn、openai-clip 和 opencv-python。Qwen 脚本另需 PyTorch、transformers、vLLM、qwen-vl-utils，以及绘图用的 matplotlib、评估用的 SciPy。作者 README 给了补充安装命令，但未锁定完整版本；实际版本应在 Linux 机器上按兼容性确定并记录。脚本还写死 `TORCH_CUDA_ARCH_LIST=8.6`，须核对实际显卡后处理。此处只是清单，**不能把 README 命令直接当作已获准安装的命令**。

## 运行前的准备顺序

1. 确认 Linux 机器的实际 GPU、显存、驱动、Python 版本、磁盘空间，以及项目根目录。确定数据是复制还是挂载；先检查 21 个视频文件夹及五个 H5 标签是否完整。
2. 在准备下载任何软件、模型或依赖前，把**实际绝对路径**逐一列给用户确认：Linux 项目根目录、Python 环境安装目录、pip/构建临时目录、模型保存目录、Hugging Face 与 Torch 缓存目录、结果目录。确认后再下载。不能用脚本默认缓存路径悄悄下载。
3. 取得参考仓库同一提交号，复制 Avenue 帧、标签及已有 CLIP 权重；检查每个视频的 JPG 数、H5 键顺序和标签长度是否相配。参考代码按排好序的视频名与 H5 顺序配对，并没有按视频名重新查找标签，顺序尤其要核对。
4. 单视频入口已经加入项目：不加 `--run` 时可在 Windows 检查本地帧与标签；真正运行时要明确给出已存在的本地 Qwen 模型路径。入口逐段保存原始回答与分数，中断后可从下一段继续。**原作者脚本没有只跑第 16 号视频的命令选项**，且处理完一整类视频才写 JSON；这些能力来自项目新入口，Linux 上的实际模型运行仍待验证。
5. 在 Linux 上先只跑 Avenue 第 16 号视频的 `bicycle`：原始 740 帧，应形成 30 个完整的 24 帧片段，即三路分数各 720 个；末尾 20 帧不参加本次分数与指标。每个片段要得到关键图、注意力图、时序图对应的三次 Qwen 回答。保存原始回答与解析后的分数，检查每个分数在 0～1、无空值，再核对标签顺序。Windows 离线检查已确认 740/30/720/20 和尾段 10 个骑车正例；真正的 Qwen 短检查通过后，才进行全量评估。
6. 全量运行五个事件：`too_close`、`bicycle`、`throwing`、`running`、`dancing`。各事件按常规口径处理 18 个视频，合计 507 个完整片段、12,168 个纳入评估的帧；五个事件共需给 2,535 个“片段×事件”组合打分，正常每组有三次 Qwen 回答。保留每类的原始三路分数、参数、日志及整理报告，不因单类失败而覆盖已成功的结果。
7. 在已有分数上先以固定参数评估：`--grid_search false --sigma 15 --alpha 0.6 --beta 0.3 --gamma 0.1`。另行复现作者默认的 `--grid_search true` 作为“测试标签选优”的参考结果，两个数字必须分别标注。作者评估脚本先合并各视频，再做高斯平滑与全局归一化；不要把它与此前 CLIP 的原始分数 AUROC 直接当作同一计算口径。

## 特别需要检查的风险

- 默认网格搜索会用**测试集真实标签**挑出最好的平滑强度和三路权重。它适合说明作者代码怎样产生最优数字，但不能当成事先固定参数的独立测试成绩。
- 原脚本只把解析后的数字写进最终 JSON，不保存 Qwen 原话；其解析函数抓取回答中出现的第一个数字。若模型回答不符合预期，可能得到错误分数。因此单视频入口应保留原话，并核对数字范围。
- 原脚本是在整类视频结束后写 JSON；中途异常可能没有可用的部分结果。正式运行前需要逐视频落盘或其他可靠的恢复办法。
- 第 16 号视频只是流程验收，不代表五类总体效果；尤其要继续关注跳舞 18 号等既有低分案例。

## 当前可执行的下一步

等待用户提供或选定 Linux GPU 环境后，先落实实际路径并请求下载位置确认；再转移现有资料，用已写好的单视频入口完成真正的 Qwen 运行，按上述 740→720 帧验收。任何新下载或安装都在路径确认**之后**进行。

来源（查看日期：2026-09-26）：

- [AnyAnomaly 官方代码与 README](https://github.com/SkiddieAhn/Paper-AnyAnomaly)
- [vLLM GPU 安装要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM Qwen2.5-VL 使用说明](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- [Qwen2.5-VL-3B-Instruct 模型页](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
