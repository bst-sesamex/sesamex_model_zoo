# SesameX Model Zoo Release Policy

本文定义 sesamex-model-zoo 的发布流程。参考 D-Robotics rdk_model_zoo 的发布规范，
按本仓现状裁剪：无 catalog 站点、单平台线、无 GitHub Release 附件校验链。

## 1. 版本

- 单产品线（C1200/A1000 双芯片型号合并交付，对应 Aura 与 Kalos 平台），使用 SemVer `MAJOR.MINOR.PATCH`
- `VERSION` 文件保存未加前缀版本号（如 `1.0.0`）
- `CHANGELOG.md` 是仓库内唯一的版本变更记录入口

## 2. 发布必改文件

每次发布提交必须更新/核对：

- `VERSION` — 版本号
- `CHANGELOG.md` — 本版变更、验证范围、已知限制
- `docs/manifests/models.yaml` — 模型资产清单（sample 路径、下载脚本、格式、sha256）
- `docs/manifests/benchmarks.yaml` — 通用 C1200 推理性能清单；设备专项原始报告另存

规则（继承 rdk_model_zoo）：

- 未知校验和必须写 `null`，**不得猜测**
- 缺失的性能/精度指标表示"本版本未测量"，**绝不表示数据受限**
- manifest 中的 sample 路径与下载脚本必须真实存在，URL 必须正确

## 3. 发布步骤

1. 核对模型资产、文件名与 SHA-256 清单一致，并核对性能清单与各 sample 文档数值一致
2. 更新 `VERSION`、`CHANGELOG.md`、两份 manifest
3. 本地跑 CI 四检查（shell/py/yaml 语法 + 敏感内容扫描）：
   ```bash
   bash -n 逐个 .sh；python3 -m py_compile 逐个 .py；
   grep 敏感模式（见 .github/workflows/ci.yml）
   ```
4. **二次审批**：按内部决策档案逐项复核（license/版权头/入库权重完整性）
5. 打 tag `v<VERSION>` 并创建 GitHub Release

## 4. 发布前强制项（对应内部决策档案 D1~D7）

- [ ] train/ 未混入任何上游框架代码（D1）
- [ ] 模型权重与 pybsnn wheel 已随仓发布且校验脚本通过（D2/D8，v1.0.1 起入库制）
- [ ] 全部源码文件版权头为 Apache-2.0 + 正确主体名；无专有声明残留（D3）
- [ ] VERSION/CHANGELOG/LICENSE 就位（D4）
- [ ] 通用性能表与 evaluator/manifest 数值一致；设备专项报告和原始日志未入算法源码仓（D5）
- [ ] 精度表只含实测数字，缺失项标注"未测量"（D6）
- [ ] sample 形态符合 CONTRIBUTING.md 七件套约定（D7）
