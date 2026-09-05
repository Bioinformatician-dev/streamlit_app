import streamlit as st

import numpy as np
import pandas as pd
import cv2
import io
import time

from PIL import Image
from ultralytics import YOLO

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Seed Phenotyping AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background: #f5f7fa;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 35px;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #064e3b,
            #047857,
            #10b981
        );
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        font-size: 18px;
        margin-top: 10px;
        opacity: 0.9;
    }

    .metric-box {
        padding: 20px;
        background: white;
        border-radius: 18px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
    }

    .metric-number {
        font-size: 32px;
        font-weight: 800;
    }

    .metric-label {
        color: #6b7280;
        font-size: 14px;
    }

    .section {
        font-size: 26px;
        font-weight: 750;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO("best.pt")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_color_features(image, mask):

    pixels = image[mask == 1]

    if len(pixels) == 0:
        return {
            "Mean R": 0,
            "Mean G": 0,
            "Mean B": 0,
            "Mean Brightness": 0
        }

    mean_rgb = pixels.mean(axis=0)

    brightness = (
        0.299 * mean_rgb[0]
        + 0.587 * mean_rgb[1]
        + 0.114 * mean_rgb[2]
    )

    return {
        "Mean R": round(float(mean_rgb[0]), 2),
        "Mean G": round(float(mean_rgb[1]), 2),
        "Mean B": round(float(mean_rgb[2]), 2),
        "Mean Brightness": round(float(brightness), 2)
    }


def analyze_seed(
    image,
    mask,
    seed_id,
    class_name,
    confidence,
    pixels_per_mm
):

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    binary = (mask > 0.5).astype(np.uint8)

    # --------------------------------------------------------
    # CONTOUR
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    contour = max(
        contours,
        key=cv2.contourArea
    )

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    area_px = cv2.contourArea(contour)

    # --------------------------------------------------------
    # PERIMETER
    # --------------------------------------------------------

    perimeter_px = cv2.arcLength(
        contour,
        True
    )

    # --------------------------------------------------------
    # MINIMUM ROTATED RECTANGLE
    # --------------------------------------------------------

    rect = cv2.minAreaRect(contour)

    center, dimensions, angle = rect

    w_px = dimensions[0]
    h_px = dimensions[1]

    length_px = max(w_px, h_px)
    width_px = min(w_px, h_px)

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    if pixels_per_mm > 0:

        length_mm = length_px / pixels_per_mm

        width_mm = width_px / pixels_per_mm

        area_mm2 = (
            area_px /
            (pixels_per_mm ** 2)
        )

        perimeter_mm = (
            perimeter_px /
            pixels_per_mm
        )

    else:

        length_mm = length_px
        width_mm = width_px
        area_mm2 = area_px
        perimeter_mm = perimeter_px

    # --------------------------------------------------------
    # CIRCULARITY
    # --------------------------------------------------------

    if perimeter_px > 0:

        circularity = (
            4 * np.pi * area_px /
            perimeter_px ** 2
        )

    else:

        circularity = 0

    circularity = min(
        max(circularity, 0),
        1
    )

    # --------------------------------------------------------
    # ASPECT RATIO
    # --------------------------------------------------------

    if width_px > 0:
        aspect_ratio = length_px / width_px
    else:
        aspect_ratio = 0

    # --------------------------------------------------------
    # EQUIVALENT DIAMETER
    # --------------------------------------------------------

    equivalent_diameter_px = np.sqrt(
        4 * area_px / np.pi
    )

    equivalent_diameter_mm = (
        equivalent_diameter_px /
        pixels_per_mm
        if pixels_per_mm > 0
        else equivalent_diameter_px
    )

    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    moments = cv2.moments(contour)

    if moments["m00"] != 0:

        cx = (
            moments["m10"] /
            moments["m00"]
        )

        cy = (
            moments["m01"] /
            moments["m00"]
        )

    else:

        cx = center[0]
        cy = center[1]

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    color = calculate_color_features(
        image,
        binary
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    data = {

        "Seed ID": seed_id,

        "Class": class_name,

        "Confidence": round(
            confidence,
            3
        ),

        "Length (mm)": round(
            length_mm,
            3
        ),

        "Width (mm)": round(
            width_mm,
            3
        ),

        "Area (mm²)": round(
            area_mm2,
            3
        ),

        "Perimeter (mm)": round(
            perimeter_mm,
            3
        ),

        "Circularity": round(
            circularity,
            3
        ),

        "Aspect Ratio": round(
            aspect_ratio,
            3
        ),

        "Equivalent Diameter (mm)": round(
            equivalent_diameter_mm,
            3
        ),

        "Centroid X": round(
            float(cx),
            2
        ),

        "Centroid Y": round(
            float(cy),
            2
        )
    }

    data.update(color)

    return data, contour


# ============================================================
# HEATMAP FUNCTION
# ============================================================

def create_heatmap(
    image,
    dataframe,
    parameter
):

    output = image.copy()

    if len(dataframe) == 0:
        return output

    height, width = image.shape[:2]

    heat = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # VALUES
    # --------------------------------------------------------

    values = dataframe[parameter].values

    if len(values) == 0:
        return output

    min_value = np.min(values)
    max_value = np.max(values)

    if max_value == min_value:

        normalized = np.ones(
            len(values)
        )

    else:

        normalized = (
            values - min_value
        ) / (
            max_value - min_value
        )

    # --------------------------------------------------------
    # PLACE GAUSSIAN SPOTS
    # --------------------------------------------------------

    for idx, row in dataframe.iterrows():

        x = int(row["Centroid X"])
        y = int(row["Centroid Y"])

        value = normalized[idx]

        cv2.circle(
            heat,
            (x, y),
            60,
            float(value),
            -1
        )

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=40
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    heat = cv2.normalize(
        heat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # --------------------------------------------------------
    # APPLY COLORMAP
    # --------------------------------------------------------

    heat_color = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET
    )

    heat_color = cv2.cvtColor(
        heat_color,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # BLEND
    # --------------------------------------------------------

    result = cv2.addWeighted(
        output,
        0.55,
        heat_color,
        0.45,
        0
    )

    return result


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🌱 Seed Phenotyping AI</h1>

        <p>
        Automated seed detection, counting, segmentation,
        morphological measurement and spatial heatmap analysis.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.40,
        step=0.05
    )

    iou = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.divider()

    st.subheader("📏 Calibration")

    calibration_method = st.radio(
        "Calibration",
        [
            "Known pixels per mm",
            "Known reference length"
        ]
    )

    if calibration_method == "Known pixels per mm":

        pixels_per_mm = st.number_input(
            "Pixels per mm",
            min_value=0.01,
            value=10.0,
            step=0.1
        )

    else:

        reference_mm = st.number_input(
            "Reference length (mm)",
            min_value=0.1,
            value=10.0,
            step=0.1
        )

        reference_pixels = st.number_input(
            "Reference length (pixels)",
            min_value=1.0,
            value=100.0,
            step=1.0
        )

        pixels_per_mm = (
            reference_pixels /
            reference_mm
        )

    st.info(
        "For real-world measurements, calibration is required. "
        "Keep the camera distance, zoom and resolution fixed."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your seed image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is None:

    st.markdown(
        """
        <div style="
            background:white;
            padding:60px;
            text-align:center;
            border-radius:20px;
            border:2px dashed #d1d5db;
        ">

        <h2>🌱 Upload a Seed Image</h2>

        <p>
        Supported setup: seeds arranged in a queue
        or spread on paper.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

image_pil = Image.open(
    uploaded_file
).convert("RGB")

image = np.array(
    image_pil
)

height, width = image.shape[:2]


# ============================================================
# MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load best.pt"
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔬 Analyze Seeds",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # ANIMATION
    # --------------------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    stages = [

        ("📷 Loading image...", 15),

        ("🧠 Running AI segmentation...", 40),

        ("🌱 Detecting individual seeds...", 60),

        ("📐 Calculating morphology...", 80),

        ("🔥 Creating spatial analysis...", 95),

        ("✅ Complete!", 100)
    ]

    for message, value in stages:

        status.info(message)

        progress.progress(value)

        time.sleep(0.25)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # RESULT IMAGE
    # --------------------------------------------------------

    annotated = image.copy()

    rows = []

    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    if result.masks is not None:

        masks = (
            result.masks.data
            .cpu()
            .numpy()
        )

        boxes = result.boxes

        names = model.names

        for i, mask in enumerate(masks):

            # Resize mask
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

            cls = int(
                boxes.cls[i]
                .cpu()
                .numpy()
            )

            conf_score = float(
                boxes.conf[i]
                .cpu()
                .numpy()
            )

            class_name = names[cls]

            # Analyze seed
            seed_data, contour = analyze_seed(
                image,
                mask,
                i + 1,
                class_name,
                conf_score,
                pixels_per_mm
            )

            if seed_data is None:
                continue

            rows.append(
                seed_data
            )

            # ------------------------------------------------
            # MASK OVERLAY
            # ------------------------------------------------

            binary = (
                mask > 0.5
            ).astype(np.uint8)

            overlay = annotated.copy()

            overlay[
                binary == 1
            ] = (
                30,
                200,
                120
            )

            annotated = cv2.addWeighted(
                annotated,
                0.70,
                overlay,
                0.30,
                0
            )

            # ------------------------------------------------
            # CONTOUR
            # ------------------------------------------------

            cv2.drawContours(
                annotated,
                [contour],
                -1,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # CENTROID
            # ------------------------------------------------

            cx = seed_data[
                "Centroid X"
            ]

            cy = seed_data[
                "Centroid Y"
            ]

            # ------------------------------------------------
            # SEED LABEL
            # ------------------------------------------------

            label = (
                f"#{i + 1} "
                f"{class_name}"
            )

            cv2.putText(
                annotated,
                label,
                (
                    int(cx),
                    int(cy)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    df = pd.DataFrame(rows)

    status.success(
        f"✅ Analysis complete — {len(df)} seeds detected."
    )

    progress.empty()

    if len(df) == 0:

        st.warning(
            "No seeds were detected. "
            "Try reducing the confidence threshold."
        )

        st.stop()

    # ========================================================
    # SAVE DATA
    # ========================================================

    st.session_state["df"] = df
    st.session_state["image"] = image
    st.session_state["annotated"] = annotated


# ============================================================
# DISPLAY RESULTS IF AVAILABLE
# ============================================================

if "df" not in st.session_state:

    st.stop()


df = st.session_state["df"]

image = st.session_state["image"]

annotated = st.session_state["annotated"]


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    '<div class="section">📊 Analysis Summary</div>',
    unsafe_allow_html=True
)

total_seeds = len(df)

avg_length = df[
    "Length (mm)"
].mean()

avg_width = df[
    "Width (mm)"
].mean()

avg_area = df[
    "Area (mm²)"
].mean()

avg_circularity = df[
    "Circularity"
].mean()


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "🌱 Total Seeds",
        total_seeds
    )


with col2:

    st.metric(
        "📏 Avg Length",
        f"{avg_length:.2f} mm"
    )


with col3:

    st.metric(
        "↔️ Avg Width",
        f"{avg_width:.2f} mm"
    )


with col4:

    st.metric(
        "📐 Avg Area",
        f"{avg_area:.2f} mm²"
    )


with col5:

    st.metric(
        "⭕ Circularity",
        f"{avg_circularity:.2f}"
    )


# ============================================================
# IMAGE VIEW
# ============================================================

st.markdown(
    '<div class="section">🔍 AI Detection</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    st.subheader("Original")

    st.image(
        image,
        use_container_width=True
    )


with col2:

    st.subheader("Segmented Seeds")

    st.image(
        annotated,
        use_container_width=True
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section">🌱 Seed Class Distribution</div>',
    unsafe_allow_html=True
)

class_counts = (
    df["Class"]
    .value_counts()
    .reset_index()
)

class_counts.columns = [
    "Class",
    "Count"
]

fig_class = px.bar(
    class_counts,
    x="Class",
    y="Count",
    text="Count",
    title="Number of Seeds per Class"
)

fig_class.update_layout(
    xaxis_title="Seed Class",
    yaxis_title="Count"
)

st.plotly_chart(
    fig_class,
    use_container_width=True
)


# ============================================================
# HEATMAP
# ============================================================

st.markdown(
    '<div class="section">🔥 Spatial Heatmap</div>',
    unsafe_allow_html=True
)

heatmap_parameter = st.selectbox(
    "Select heatmap parameter",
    [
        "Area (mm²)",
        "Length (mm)",
        "Width (mm)",
        "Circularity",
        "Aspect Ratio",
        "Confidence",
        "Mean Brightness"
    ]
)

heatmap_image = create_heatmap(
    image,
    df,
    heatmap_parameter
)

st.image(
    heatmap_image,
    caption=f"Spatial heatmap: {heatmap_parameter}",
    use_container_width=True
)


# ============================================================
# SCATTER PLOT
# ============================================================

st.markdown(
    '<div class="section">📈 Morphological Relationships</div>',
    unsafe_allow_html=True
)

x_axis = st.selectbox(
    "X-axis",
    [
        "Length (mm)",
        "Width (mm)",
        "Area (mm²)",
        "Perimeter (mm)",
        "Circularity",
        "Aspect Ratio"
    ],
    index=0
)

y_axis = st.selectbox(
    "Y-axis",
    [
        "Area (mm²)",
        "Length (mm)",
        "Width (mm)",
        "Perimeter (mm)",
        "Circularity",
        "Aspect Ratio"
    ],
    index=0
)

fig_scatter = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color="Class",
    hover_data=[
        "Seed ID",
        "Confidence"
    ],
    title=f"{x_axis} vs {y_axis}"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# ============================================================
# DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section">📊 Measurement Distribution</div>',
    unsafe_allow_html=True
)

distribution_parameter = st.selectbox(
    "Measurement",
    [
        "Length (mm)",
        "Width (mm)",
        "Area (mm²)",
        "Perimeter (mm)",
        "Circularity",
        "Aspect Ratio"
    ]
)

fig_distribution = px.histogram(
    df,
    x=distribution_parameter,
    color="Class",
    marginal="box",
    title=f"Distribution of {distribution_parameter}"
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)


# ============================================================
# INDIVIDUAL SEED ANALYSIS
# ============================================================

st.markdown(
    '<div class="section">🔬 Individual Seed Analysis</div>',
    unsafe_allow_html=True
)

available_classes = sorted(
    df["Class"].unique()
)

selected_classes = st.multiselect(
    "Filter seed classes",
    available_classes,
    default=available_classes
)

filtered_df = df[
    df["Class"].isin(
        selected_classes
    )
]

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CLASS STATISTICS
# ============================================================

st.markdown(
    '<div class="section">📋 Class-wise Statistics</div>',
    unsafe_allow_html=True
)

numeric_columns = [
    "Length (mm)",
    "Width (mm)",
    "Area (mm²)",
    "Perimeter (mm)",
    "Circularity",
    "Aspect Ratio"
]

class_statistics = (
    df.groupby("Class")[numeric_columns]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    )
)

st.dataframe(
    class_statistics,
    use_container_width=True
)


# ============================================================
# EXCEL EXPORT
# ============================================================

st.markdown(
    '<div class="section">📥 Export Results</div>',
    unsafe_allow_html=True
)


# CSV
csv_data = df.to_csv(
    index=False
).encode("utf-8")


# Excel
excel_buffer = io.BytesIO()

with pd.ExcelWriter(
    excel_buffer,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        index=False,
        sheet_name="Seed Measurements"
    )

    class_statistics.to_excel(
        writer,
        sheet_name="Class Statistics"
    )


col1, col2, col3 = st.columns(3)


with col1:

    st.download_button(
        "📄 Download CSV",
        data=csv_data,
        file_name="seed_measurements.csv",
        mime="text/csv",
        use_container_width=True
    )


with col2:

    st.download_button(
        "📊 Download Excel",
        data=excel_buffer.getvalue(),
        file_name="seed_phenotyping.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )


# ============================================================
# ANNOTATED IMAGE DOWNLOAD
# ============================================================

buffer = io.BytesIO()

Image.fromarray(
    annotated
).save(
    buffer,
    format="PNG"
)


with col3:

    st.download_button(
        "🖼️ Download Annotated Image",
        data=buffer.getvalue(),
        file_name="seed_analysis.png",
        mime="image/png",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#6b7280;
        padding:20px;
    ">

    🌱 <b>Seed Phenotyping AI</b><br>

    YOLO Segmentation • OpenCV • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)
