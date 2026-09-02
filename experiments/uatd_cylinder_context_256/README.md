# UATD cylinder target-context, 256×256×1

This experiment uses the UATD cylinder instances (class id `8`) instead of
the small KLSG set. Each sample is a square target-context window centered on
the annotated cylinder. The default source window is five times the longer
side of the target box, then it is resized to `256×256` and saved as one
grayscale channel. Border pixels use edge replication when the window reaches
the image boundary.

UATD provides frame-level range, azimuth, elevation, sound speed, and
frequency metadata, but it does not provide a pixel-level acoustic-shadow
mask. The generated data therefore preserves possible shadow/context pixels
without claiming that every patch contains a labelled shadow.

## Build the dataset

```powershell
$PYTHON = "C:\Users\Yanshier\.conda\envs\sg2\python.exe"
$UATD = "C:\C2\Document\sonar_research\dataset\Sonar_Dataset\UATD"

& $PYTHON tools\sonar\make_uatd_cylinder_patches.py `
  --uatd-root $UATD `
  --out experiments\uatd_cylinder_context_256\dataset_uatd_context\train `
  --contact-sheet experiments\uatd_cylinder_context_256\uatd_cylinder_context_preview.png

& $PYTHON dataset_tool.py `
  --source experiments\uatd_cylinder_context_256\dataset_uatd_context\train `
  --dest experiments\uatd_cylinder_context_256\uatd_cylinder_context_256.zip `
  --width 256 --height 256 --resize-filter lanczos
```

The manifest is kept beside the generated PNGs and records the source box,
patch box, context scale, and UATD sonar metadata. Keep the contact sheet
outside the source directory because `dataset_tool.py` recursively ingests
image files.

## Start training

On the current RTX 4060 8 GB setup, the generated UATD zip completed a real
training smoke tick at one GPU, batch size `8`, and 256 resolution without
CUDA OOM (peak reported GPU memory was about 4.93 GiB):

```powershell
& $PYTHON train.py `
  --outdir experiments\uatd_cylinder_context_256\gan_runs `
  --data experiments\uatd_cylinder_context_256\uatd_cylinder_context_256.zip `
  --gpus 1 --batch 8 --cfg auto --aug noaug --metrics none
```

If a full run reports CUDA OOM, regenerate at `128×128` and use batch `16`;
the current repository already handles one-channel datasets automatically.
No SSS-specific slant-range warp or uncalibrated acoustic simulation is
enabled in this baseline.
