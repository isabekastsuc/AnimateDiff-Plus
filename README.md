# AnimateDiff

This repository is a vibe-coded modification of [AnimateDiff](https://arxiv.org/abs/2307.04725) [ICLR2024 Spotlight] that switches the foundation to the Flux.2 Klein model series for fast i2i capabilities instead of morphing backgrounds.

## Location-first workflow

The original AnimateDiff path in this repo still generates whole clips from a single denoising process, which means backgrounds and characters drift together. A new location-first workflow now exists for the stable-room use case:

- Cache or load a canonical scene plate once.
- Generate or ingest foreground character frames separately.
- Build masks from alpha, chroma key, or lightness.
- Composite the moving character onto the stable scene plate.

The new entrypoint is:

```bash
python scripts/animate_location.py --config configs/prompts/4_location_composite/4_1_location_composite_demo.yaml
```

Base install:

```bash
pip install -r requirements.txt
```

Optional acceleration:

```bash
pip install -r requirements-optional.txt
```

`xformers` is optional. The repo will try to enable it when present, but falls back cleanly if it is unavailable or if the installed wheel does not match the active PyTorch/CUDA build.

Config sections:

- `models`: base model, motion module, optional LoRA/controlnet for the foreground generation pass
- `scene`: canonical location image or prompt-backed cached scene plate
- `foreground`: either `frames_path`/`masks_path` or a prompt that generates an isolated character clip
- `composite`: mask extraction, placement, shadow, frame count, fps
- `outputs`: output directory and filenames

If `scene.image_path` is omitted, `scene.prompt` will be generated once, stored under `scene.cache_dir`, and reused on later runs. If `foreground.frames_path` is omitted, the script uses the existing AnimateDiff pipeline to generate a foreground clip, then composites it onto the cached location image.
