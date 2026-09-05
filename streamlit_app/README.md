# Lentil Seed Phenotyping Station — Streamlit edition

Same tool as the Flask version, rebuilt in Streamlit. `pipeline.py` is
shared byte-for-byte between both — only the upload/ruler/results
interface is different.

## Run it

```bash
cd streamlit_app
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens automatically at http://localhost:8501

## A version note on the ruler tool

The ruler/manual-measurement tool uses `streamlit-drawable-canvas`, which
still relies on Streamlit's older component API. Newer Streamlit releases
(1.4x+) dropped support for it, so **`requirements.txt` pins an exact,
tested combination**: `streamlit==1.28.0` with
`streamlit-drawable-canvas==0.9.3`. Installing the two mismatched (e.g.
upgrading Streamlit alone) will break the canvas with a
`StreamlitAPIException` about component registration — install from this
`requirements.txt` as-is rather than freeform.

If you outgrow this and want a current Streamlit, the tradeoff is
replacing the canvas ruler with a coordinate-entry form (type in two pixel
coordinates instead of drawing a line) — ask if you want that swapped in.

## Same detection-mode behaviour as the Flask version

Drop a trained `best.pt` into `models/` for YOLO detection with the 6 seed
classes; without it, the app uses the classical adaptive-threshold +
watershed fallback.

## Same scope note

This covers the measurement pipeline (upload, calibrate, detect, measure,
export). The notebook's research/QA panels — reliability/repeatability,
EFD+PCA shape atlas, caliper validation — aren't in this tool; run those
in the notebook.
