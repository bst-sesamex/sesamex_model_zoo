# OpenStereo-2 model files

Both platform packages are rebuilt from the repository's own float ONNX
(`conversion/assets/lightstereo/checkpoint_epoch_299.onnx`) with BSTNNX 8.3.1,
so the C1200 and Kalos (A1000) packages share the same graph topology and
output definition: a single output tensor `disp_pred [960,544,1,1]`
(normalized disparity 0..1; the C1200 package packs it as INT16 with
dequant scale 1/32768).

The dual-softmax consensus (two `[9,960,544,1]` logits tensors) of earlier
published packages was fused into the network by the `fuse_mul_reducesum`
graph-optimization pass; the demos in `runtime/` were updated accordingly
(single-output post-processing, `× max_disp` to restore pixel disparity).

- `lightstereo-c1200`: `lightstereo.ModelGraph.bin`, `lightstereo.ModelGraph.bin.txt`, `lightstereo.meta`, `lightstereo.sdag.bin`, `weights.bin`
- `lightstereo-a1000`: same five-file layout as `lightstereo-c1200`

## Aura C1200 inference performance

Latency is the median of three post-warmup C++ run+wait calls measured with a
board-side monotonic clock. FPS = 1000 / latency; pre/postprocessing is excluded.

| Variant | FPS | Latency |
|---|---:|---:|
| `lightstereo` | 4.59 | 217.787 ms (cpp) |

## Kalos (A1000 chip)

`lightstereo-a1000/` ships with the repository (same five-file layout).

- Consistency: `bsnn_sample_internal -c` reports bit-exact match against the
  package's `ref_output`.
- Latency: ~232.8 ms (py) / ~214.1 ms (cpp) per inference on the validated board.
- The first-inference ~8 s timeout reported with the superseded package is
  resolved by this rebuilt package.
