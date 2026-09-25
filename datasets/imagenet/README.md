# ImageNet Dataset Resources

ImageNet-1k 验证集用于图像分类 Top-1/Top-5 评测。ImageNet 数据需从官方渠道
（https://image-net.org）申请下载，本目录只提供类别标签文件与目录布局说明。

## 1. 目录布局（评测脚本预期）

```text
imagenet/
├── README.md
├── imagenet_classes.txt       # 1000 类标签（按类号排序）
└── val_images/                # 用户自行下载解压（50,000 张）
    ├── ILSVRC2012_val_00000001.JPEG
    └── ...
```

## 2. 被引用的 sample

- samples/vision/classification（Top-1/Top-5）
