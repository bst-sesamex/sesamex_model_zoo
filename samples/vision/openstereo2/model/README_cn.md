# OpenStereo-2 模型文件

两平台的模型包均以本仓库自带的浮点 ONNX
（`conversion/assets/lightstereo/checkpoint_epoch_299.onnx`）经 BSTNNX 8.3.1
重新编译，因此 C1200 与 Kalos（A1000）两包共享同一图拓扑与输出定义：
单一输出张量 `disp_pred [960,544,1,1]`（归一化视差 0..1；C1200 包以 INT16
打包、反量化系数 1/32768）。

早期发布包中的双 softmax 共识（两个 `[9,960,544,1]` logits 张量）已被
`fuse_mul_reducesum` 图优化 pass 融合进网络；`runtime/` 下的 demo 已同步
适配（单输出后处理，`× max_disp` 还原像素视差）。

- `lightstereo-c1200`：`lightstereo.ModelGraph.bin`, `lightstereo.ModelGraph.bin.txt`, `lightstereo.meta`, `lightstereo.sdag.bin`, `weights.bin`
- `lightstereo-a1000`：与 c1200 相同的五文件布局

## Aura C1200 推理性能

耗时为预热后 C++ run+wait 三轮板端单调时钟中位数；FPS = 1000 / 耗时，不含前后处理。

| 模型 | FPS | 耗时 |
|---|---:|---:|
| `lightstereo` | 4.59 | 217.787 ms (cpp) |

## Kalos（A1000 芯片）

`lightstereo-a1000/` 随仓库发布（与 c1200 相同的五文件布局）。

- 一致性：`bsnn_sample_internal -c` 对包内 `ref_output` 报告 bit-exact match。
- 耗时：验证板上约 232.8 ms（py）/ 214.1 ms（cpp）每次推理。
- 旧 Enrsd 包曾存在的"首次推理 ~8 秒超时"已知问题，随本次重编译包解决。
