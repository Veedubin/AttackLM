# DeepSpeed Configuration Presets

Pre-built DeepSpeed ZeRO configs for common hardware setups.

## Presets

| File | ZeRO Stage | CPU Offload | Best For |
|------|-----------|-------------|----------|
| `zero3_cpu_offload.json` | 3 | Yes (params + optimizer) | Single GPU, model > VRAM. 40B+ on 16GB GPU + 64GB RAM |
| `zero3_gpu_only.json` | 3 | No | Multi-GPU, model fits across GPUs |
| `zero2_cpu_offload.json` | 2 | Yes (optimizer only) | Single GPU, model ~2x VRAM. Faster than ZeRO-3 |

## Usage

```bash
# Use a preset config
attacklm train -- --use-deepspeed --deepspeed-config presets/deepspeed/zero3_cpu_offload.json --dataset data/balanced.jsonl --train

# Auto-generate (defaults to ZeRO-3 + CPU offload)
attacklm train -- --use-deepspeed --dataset data/balanced.jsonl --train

# ZeRO-2 for faster training on models that almost fit
attacklm train -- --use-deepspeed --deepspeed-stage 2 --dataset data/balanced.jsonl --train
```

## Hardware Reference

| GPU VRAM | System RAM | Recommended Config | Max Model Size |
|----------|-----------|-------------------|----------------|
| 8 GB | 32 GB | zero2_cpu_offload | ~13B |
| 16 GB | 64 GB | zero3_cpu_offload | ~40B |
| 24 GB | 64 GB | zero3_cpu_offload | ~70B |
| 24 GB | 128 GB | zero3_cpu_offload | ~70B+ |
| 2× 24 GB | 64 GB | zero3_gpu_only | ~70B |