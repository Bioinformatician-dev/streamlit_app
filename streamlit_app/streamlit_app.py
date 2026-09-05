"""
Lentil Seed Phenotyping Station — Streamlit edition.

Same measurement pipeline as the Flask version (pipeline.py is shared,
byte for byte). This file only handles upload, the ruler / manual
measurement tool, running the analysis, and rendering results + exports.
"""
import base64
import io
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from streamlit_drawable_canvas import st_canvas

import pipeline

BASE_DIR = Path(__file__).parent
JINJA_ENV = Environment(loader=FileSystemLoader(BASE_DIR / "templates"))

st.set_page_config(page_title="Lentil Phenotyping Station", page_icon="🌱", layout="wide")


# --------------------------------------------------------------------------
# Look and feel
# --------------------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"]  { font-family:'IBM Plex Sans', sans-serif; }
    .stApp { background:#F3EFE3; }
    h1, h2, h3 { font-family:'Fraunces', serif !important; color:#153328; }
    .block-container { padding-top: 1.2rem; }

    .lp-hero{
      background:radial-gradient(120% 140% at 15% 0%, #1C4433 0%, #0F231B 65%);
      color:#F3EFE3; border-radius:14px; padding:34px 36px; margin-bottom:22px;
    }
    .lp-hero h1{ color:#F3EFE3 !important; font-size:2.1rem; margin:6px 0 10px; }
    .lp-hero p{ color:#E7E1CE; max-width:60ch; margin:0; }
    .lp-brand{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; color:#E7E1CE; letter-spacing:0.04em; }

    .lp-pill{
      display:inline-flex; align-items:center; gap:8px; font-family:'IBM Plex Mono', monospace;
      font-size:0.76rem; padding:6px 12px; border-radius:999px; border:1px solid rgba(243,239,227,0.25);
      margin-top:14px; background:rgba(255,255,255,0.05);
    }
    .lp-dot{ width:8px; height:8px; border-radius:50%; }

    .stButton>button, .stDownloadButton>button{
      background:#153328; color:#F3EFE3; border-radius:7px; border:1px solid #153328;
      font-weight:600;
    }
    .stButton>button:hover, .stDownloadButton>button:hover{ background:#1C4433; color:#F3EFE3; }

    div[data-testid="stMetric"]{
      background:#153328; border-radius:10px; padding:10px 6px;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div{ color:#F3EFE3 !important; }
    </style>
    """, unsafe_allow_html=True)


def hero(model_ready):
    dot_color = "#6FAF9E" if model_ready else "#E08D3C"
    pill_text = ("YOLO seed classifier loaded" if model_ready else
                 "Running in classical-vision mode — drop best.pt into models/ for class labels")
    st.markdown(f"""
    <div class="lp-hero">
      <div class="lp-brand">🌱 LENTIL PHENOTYPING STATION</div>
      <h1>See every seed. Measure the whole tray.</h1>
      <p>Upload a tray photo, calibrate a ruler, and get per-seed size, shape,
      colour and coat-pattern measurements — annotated images, CSV and an
      HTML report, all from one page.</p>
      <div class="lp-pill"><span class="lp-dot" style="background:{dot_color}"></span>{pill_text}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def decode_upload(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def png_b64(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def line_from_canvas(objects):
    """Fabric.js line object -> absolute (x1, y1, x2, y2) in canvas-display space."""
    if not objects:
        return None
    obj = objects[-1]  # most recently drawn line
    if obj.get("type") != "line":
        return None
    sx, sy = obj.get("scaleX", 1), obj.get("scaleY", 1)
    x1 = obj.get("left", 0) + obj.get("x1", 0) * sx
    y1 = obj.get("top", 0) + obj.get("y1", 0) * sy
    x2 = obj.get("left", 0) + obj.get("x2", 0) * sx
    y2 = obj.get("top", 0) + obj.get("y2", 0) * sy
    return x1, y1, x2, y2


def build_csv_bundle(df_seeds, stats):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        export_cols = [c for c in df_seeds.columns if not c.startswith("_")]
        zf.writestr("lentil_per_seed.csv",
                    df_seeds[export_cols].to_csv(index=False) if not df_seeds.empty else "")
        zf.writestr("lentil_statistics.csv", stats.to_csv() if not stats.empty else "")
        if not df_seeds.empty:
            per_image = df_seeds.groupby("image").agg(seed_count=("seed_id", "count")).reset_index()
            zf.writestr("lentil_per_image.csv", per_image.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


def build_html_report(df_seeds, stats, images, mode, px_per_mm):
    export_cols = [c for c in df_seeds.columns if not c.startswith("_")]
    seeds_table = df_seeds[export_cols].round(4) if not df_seeds.empty else pd.DataFrame()
    tmpl = JINJA_ENV.get_template("report.html")
    return tmpl.render(
        generated=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        mode=mode,
        units="mm" if px_per_mm else "pixels",
        px_per_mm=px_per_mm,
        n_images=len(images),
        n_seeds=int(len(df_seeds)),
        images=images,
        stat_rows=(stats.reset_index().rename(columns={"index": "trait"})
                   .round(4).to_dict(orient="records")) if not stats.empty else [],
        seed_columns=list(seeds_table.columns),
        seed_rows=seeds_table.to_dict(orient="records"),
    )


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
inject_css()
model_ready = pipeline.MODEL_PATH.exists()
hero(model_ready)

if "px_per_mm" not in st.session_state:
    st.session_state.px_per_mm = None
if "results" not in st.session_state:
    st.session_state.results = None

# ---- Step 1: upload ----
st.markdown("### 1 · Upload tray photos")
uploaded_files = st.file_uploader(
    "One image or a whole batch. JPG or PNG.",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
)
if uploaded_files:
    st.image([f for f in uploaded_files], width=110,
              caption=[f.name for f in uploaded_files])

# ---- Step 2: ruler ----
st.markdown("### 2 · Calibrate the ruler")
st.caption(
    "Draw a line across a known distance — a ruler tick, a coin, a graph-paper "
    "square — then enter the real distance in millimetres. Skip this to get "
    "measurements in pixels only. Switch to manual measurement to check any "
    "distance in the image without setting the scale."
)

if uploaded_files:
    calib_col, ctrl_col = st.columns([1.4, 1])
    ref_file = uploaded_files[0]
    pil_img = Image.open(ref_file).convert("RGB")
    ow, oh = pil_img.size
    display_w = min(700, ow)
    scale_factor = ow / display_w
    display_h = int(oh / scale_factor)
    display_img = pil_img.resize((display_w, display_h))

    with calib_col:
        canvas_result = st_canvas(
            fill_color="rgba(224, 141, 60, 0.0)",
            stroke_width=3,
            stroke_color="#E08D3C",
            background_image=display_img,
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="line",
            key="ruler_canvas",
        )

    with ctrl_col:
        mode = st.radio("Mode", ["Calibrate scale", "Manual measurement"], horizontal=False)
        objects = (canvas_result.json_data or {}).get("objects", []) if canvas_result else []
        line = line_from_canvas(objects)

        if line:
            x1, y1, x2, y2 = line
            px_dist_display = float(np.hypot(x2 - x1, y2 - y1))
            px_dist_full = px_dist_display * scale_factor
        else:
            px_dist_full = None

        if mode == "Calibrate scale":
            real_mm = st.number_input("Real-world distance (mm)", min_value=0.0, step=0.1, value=0.0)
            if px_dist_full:
                st.caption(f"Line drawn: {px_dist_full:.1f} px (full resolution)")
            if st.button("Set scale", disabled=not (px_dist_full and real_mm > 0)):
                st.session_state.px_per_mm = px_dist_full / real_mm
            if st.session_state.px_per_mm:
                st.success(f"Scale set: {st.session_state.px_per_mm:.3f} px/mm")
            else:
                st.info("No scale set — results will be in pixels.")
        else:
            if px_dist_full:
                txt = f"Line drawn: {px_dist_full:.1f} px"
                if st.session_state.px_per_mm:
                    txt += f"  ({px_dist_full / st.session_state.px_per_mm:.2f} mm)"
                st.info(txt)
            else:
                st.caption("Draw a line on the image to measure it.")
else:
    st.caption("Upload an image first.")

# ---- Step 3: run ----
st.markdown("### 3 · Detect, segment and measure")
st.caption("Runs seed detection, contour segmentation, morphology, colour and coat-pattern analysis on every uploaded image.")

run_clicked = st.button("Run analysis", disabled=not uploaded_files, type="primary")

if run_clicked:
    with st.spinner("Measuring seeds…"):
        all_rows, images_out, mode_used = [], [], "classical"
        for f in uploaded_files:
            img = decode_upload(f)
            if img is None:
                continue
            df, vis, mode_used = pipeline.analyze_image(img, f.name, px_per_mm=st.session_state.px_per_mm)
            if not df.empty:
                all_rows.append(df)
            images_out.append({
                "name": f.name,
                "seed_count": 0 if df.empty else int(len(df)),
                "annotated_png_b64": png_b64(vis),
            })
        df_seeds = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
        stats = pipeline.compute_statistics(df_seeds) if not df_seeds.empty else pd.DataFrame()
        st.session_state.results = {
            "df_seeds": df_seeds, "stats": stats, "images": images_out,
            "mode": mode_used, "px_per_mm": st.session_state.px_per_mm,
        }

# ---- Step 4: results ----
if st.session_state.results:
    res = st.session_state.results
    df_seeds, stats, images_out = res["df_seeds"], res["stats"], res["images"]

    st.markdown("### 4 · Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images analysed", len(images_out))
    m2.metric("Seeds measured", int(len(df_seeds)))
    m3.metric("Units", "mm" if res["px_per_mm"] else "pixels")
    m4.metric("Detection mode", "YOLO" if res["mode"] == "yolo" else "classical CV")

    cols = st.columns(min(3, max(1, len(images_out))))
    for i, im in enumerate(images_out):
        with cols[i % len(cols)]:
            st.image(base64.b64decode(im["annotated_png_b64"]),
                     caption=f"{im['name']} — {im['seed_count']} seeds")

    st.markdown("#### Per-seed measurements")
    if not df_seeds.empty:
        export_cols = [c for c in df_seeds.columns if not c.startswith("_")]
        st.dataframe(df_seeds[export_cols].round(4), use_container_width=True, height=280)
    else:
        st.caption("No seeds measured.")

    st.markdown("#### Statistics")
    if not stats.empty:
        st.dataframe(stats, use_container_width=True)
    else:
        st.caption("Nothing to summarise yet.")

    # ---- Step 5: export ----
    st.markdown("### 5 · Export")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download CSV bundle",
            data=build_csv_bundle(df_seeds, stats),
            file_name="lentil_results.zip",
            mime="application/zip",
        )
    with e2:
        html_report = build_html_report(df_seeds, stats, images_out, res["mode"], res["px_per_mm"])
        st.download_button(
            "Download HTML report",
            data=html_report.encode("utf-8"),
            file_name="lentil_report.html",
            mime="text/html",
        )

st.markdown(
    "<p style='text-align:center;color:#8a8468;font-family:IBM Plex Mono,monospace;"
    "font-size:0.8rem;margin-top:40px;'>Ported from the lentil phenotyping notebook"
    " · classical-vision fallback when no trained weights are loaded</p>",
    unsafe_allow_html=True,
)
