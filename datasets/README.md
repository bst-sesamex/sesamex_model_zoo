# Datasets

数据集准备脚本目录。**数据本体一律不入仓**：每个数据集一个子目录，提供
README（布局说明）+ 下载脚本（或官方申请指引）+ 类别标签文件。

| 数据集 | 用途 | 被引用的 sample |
|---|---|---|
| [coco/](./coco/) | 检测/分割/姿态 mAP | yolov8, yolov12, yolov5, fcos |
| [imagenet/](./imagenet/) | 分类 Top-1/Top-5 | classification |
| tusimple/ | 车道线 F1（需官方申请） | condlanenet |
| sceneflow/ | 立体匹配 EPE（需官方申请） | openstereo2 |

TuSimple 与 SceneFlow 受数据集许可约束，需从官方渠道获取；此处仅记录目录布局约定
（`tusimple/test_set/`、`sceneflow/StereoMatching/`，与各 sample evaluator 文档口径一致）。
