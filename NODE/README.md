# NODE

Local hosted super-resolution website for this workspace.

This app serves a single-page ROI visualizer and runs prediction locally with the checkpoint files already present in the repo root. The available weights in this workspace are SRCNN checkpoints, so the backend uses a 4-channel SRCNN model for inference and shows the bicubic baseline alongside it.

## Run

From this folder:

```bash
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Flow

1. Upload an image.
2. Generate the local SR preview.
3. Choose a box size.
4. Click on either preview to place the ROI.
5. Compare bicubic, SR, and absolute difference in the same region.