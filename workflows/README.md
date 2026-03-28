# ComfyUI Workflows

Pre-configured workflows tested on a single AMD MI50 32GB with `noflash-attention`.

## Workflows

### Wan22-5B-MI50.json
- **Model:** Wan 2.2 Text-to-Video 5B (FP16)
- **Resolution:** 832x480
- **Duration:** 2.5s (41 frames @ 16fps)
- **Benchmark time:** ~5 minutes
- **VRAM peak:** ~17.6 GB
- **DRAM usage:** ~15 GB

### LTX23-22B-MI50.json
- **Model:** LTX-2.3 22B Distilled (GGUF Q6_K)
- **Resolution:** 1280x704
- **Duration:** 5.2s (129 frames @ 25fps) with audio
- **Benchmark time:** ~20 minutes
- **VRAM peak:** ~20 GB
- **DRAM usage:** ~30 GB+ (CPU VAE decode + model offloading)

## Requirements

- `noflash-attention` installed (`pip install noflash-attention`)
- ComfyUI with the `noflash-attention` custom node in `custom_nodes/`
- **VRAM:** 32 GB minimum (MI50, Radeon VII)
- **DRAM:** 64 GB minimum, 128 GB recommended — ComfyUI offloads model weights to system RAM when VRAM is full, and VAE decode runs on CPU for the LTX workflow
- Model weights downloaded to your ComfyUI `models/` directory

## Usage

1. Drag the `.json` file onto the ComfyUI canvas
2. Update model paths if they don't match your local setup
3. Queue the prompt

## Notes

- These workflows use `--force-fp16` launch flag. Ensure ComfyUI is started with it.
- Setting `MIOPEN_FIND_MODE=FAST` is recommended for AMD GPUs (avoids slow MIOpen solver search on first run).
- The LTX workflow uses optimized VAE tiling (spatial_tiles=4, temporal_tile_length=4) to prevent OOM during decode. Adjust if your VRAM differs.
- Benchmarks are at 150W GPU power limit. Higher power limits will be faster.
