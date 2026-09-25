# Model Zoo 板端构建与计时

本仓库的模型文件和 C++ 示例以原工程
`bst_robot_model_zoo_new_bsnn/bst_robot_model_zoo_new_bsnn` 为基准。本说明只定义通用构建与计时方法。

## 目录和入口

- C++ 源码：`samples/vision/<sample>/runtime/cpp/src/`，可执行文件：同目录的 `prebuild/app/`。
- Python 示例：`samples/vision/<sample>/runtime/python/pybsnn*.py`。
- 模型文件：`samples/vision/<sample>/model/c1200/<variant>-c1200/`；保留原工程的 `.meta`、`*_stripped.lib` 或 `.sdag.bin` 文件名。
- 构建：`python3 utils/tools/build_all_models_native.py`（10 个 C++ 工程）。
- C++ 汇总：`bash utils/tools/run_all_models_benchmark.sh --device C1200`。
- Python 汇总：`bash utils/tools/run_all_pybsnn_benchmark.sh --device C1200`（12 个示例）。

三个 Python 入口都支持 `--dry-run`，先用它核对将执行的路径。`build_all_models_native.py`
设置 `MODEL_ZOO_HOST_NATIVE=1`，在每个 `runtime/cpp/build/` 做干净构建，并把新二进制安装到
`runtime/cpp/prebuild/app/`。该动作会覆盖同名旧二进制；报告性能前必须确认构建日志中全部工程成功。
模型包的 `model/download.sh` 仅检查文件在位，不能代替推理和精度验收。

## 统一计时口径

1. 加载模型后预热一次；C++ 在有效输入复制、同步后预热，Python 在输入预处理后预热。
2. 使用**板端单调时钟**计时一次同步推理调用：C++ 包住 `bsnn_run` 与 `bsnn_wait`，Python 包住 `BSNNLite.inference`。
3. `infer_ms` 不含模型加载、显式的输入预处理和后处理；C++ 的输入/输出 `mem_sync` 在计时区间外。Python 绑定内部开销仍属于 `BSNNLite.inference`，因此两种 API 的数值可能不同。
4. 同一行的 `FPS = 1000 / infer_ms`。没有直接打印 `bsnn model inference time` 的任务显示缺失并计入失败，不从 FPS 反推耗时。
5. `wall_s` 是整次子进程的启动到退出时间，用于诊断，不当作单次推理耗时。非零退出码（包括推理后析构崩溃）计入失败。

性能报告应在正确性通过后采集：固定模型文件 SHA、runtime/driver/firmware、引擎与核心掩码、频率和温度条件，每个配置至少三轮；记录原始日志、boot ID、每轮耗时与统计值。一次 demo 成功只能作为烟测，不应更新首页历史性能表。

## 板端命令

```bash
cd /path/to/sesamex-model-zoo

# 按样本校验模型包文件
for d in samples/vision/*/model; do bash "$d/download.sh"; done

# 编译 C++，然后执行 C++ 示例
python3 utils/tools/build_all_models_native.py
bash utils/tools/run_all_models_benchmark.sh --device C1200 --no-continue-on-error

# 用仓库内 wheel 安装到独立目录，避免更改系统 Python 包
python3 -m pip install --no-deps --target ./pydeps pybsnn/aura/*.whl
PYTHONPATH="$PWD/pydeps" python3 utils/tools/run_all_pybsnn_benchmark.py --device C1200
```

批量运行前先独立验证高风险或新包模型；单个 NPU 超时后应保存日志、boot ID 与驱动信息，并停止后续计时，避免污染其余结果。
