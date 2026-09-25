# COCO Dataset Resources

COCO（Common Objects in Context）是目标检测/实例分割/关键点评测的通用数据集。
本目录提供类别标签文件与完整数据集自动下载脚本；数据本体不入仓。

## 1. 目录结构

```text
coco/
├── README.md                  # 本说明
├── download_full_coco.sh      # 下载完整 COCO val2017 脚本
└── coco_classes.names         # 80 类标签（与模型 class_id 顺序一致）
```

## 2. 使用

```bash
bash download_full_coco.sh          # 默认下载到 ./val2017/
python3 -m pycocotools              # 评测需安装 pycocotools
```

## 3. 被引用的 sample

- samples/vision/yolov8（检测/分割/姿态）
- samples/vision/yolov12、yolov5、fcos（检测）
