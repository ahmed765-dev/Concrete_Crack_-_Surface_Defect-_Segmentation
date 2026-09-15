import os
import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crack Segmentation AI | YOLOv8",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------
DEFAULT_MODEL_PATH = "v4.pt"

@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    if not os.path.exists(weights_path):
        return None
    try:
        model = YOLO(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading model weights from {weights_path}: {e}")
        return None

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Model Settings")
    st.caption("Crack Segmentation & Detection Parameters")

    model_file = st.text_input(
        "Model Weights Path",
        value=DEFAULT_MODEL_PATH,
        help="Path to the trained YOLO segmentation weights file (e.g., v4.pt)"
    )

    confidence_slider = st.slider(
        "Confidence Threshold (conf)",
        min_value=0.01,
        max_value=1.00,
        value=0.15,
        step=0.01,
        help="Minimum confidence threshold for crack detection masks."
    )

    st.markdown("---")
    st.subheader("Model Inference Parameters")
    st.markdown(
        f"""
        - **Image Size (`imgsz`):** `1024`
        - **Confidence (`conf`):** `{confidence_slider:.2f}`
        - **IoU Threshold (`iou`):** `0.25`
        - **Max Detections (`max_det`):** `20`
        - **Retina Masks:** `True` (high-res contours)
        - **Augmentation (`augment`):** `False`
        """
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Upload high-resolution images of concrete, pavement, or wall surfaces for best crack segmentation results.")

# ---------------------------------------------------------
# Main UI & Inference
# ---------------------------------------------------------
st.markdown('<div class="main-title">🔍 Surface Crack Detection & Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated crack inspection using fine-tuned YOLO segmentation model</div>', unsafe_allow_html=True)

with st.spinner("Loading model weights..."):
    model = load_yolo_model(model_file)

if model is None:
    st.warning(
        f"⚠️ Weights file `{model_file}` was not found in the current directory.\n\n"
        "Please ensure `v4.pt` is placed in the project folder, or provide the correct path in the sidebar."
    )

uploaded_file = st.file_uploader(
    "Choose an image for crack analysis (JPG, JPEG, PNG)...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a concrete wall, bridge deck, pavement, or building surface."
)

if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != (uploaded_file.name if uploaded_file else None):
    st.session_state["analyzed"] = False
    st.session_state["last_uploaded"] = uploaded_file.name if uploaded_file else None

if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file)
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            run_btn = st.button("🚀 Analyze Cracks", type="primary")

        if run_btn:
            st.session_state['analyzed'] = True

        if st.session_state.get('analyzed', False):
            if model is None:
                st.error("Cannot perform inference: Model weights not loaded.")
            else:
                with st.spinner("Processing image through YOLO segmentation model..."):
                    results = model.predict(
                        source=input_image,
                        imgsz=1024,
                        conf=confidence_slider,
                        iou=0.25,
                        max_det=20,
                        retina_masks=True,
                        augment=False
                    )

                    annotated_bgr = results[0].plot(labels=False, conf=False)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    boxes = results[0].boxes
                    masks = results[0].masks
                    detected_count = len(boxes) if boxes is not None else 0
                    speed_info = results[0].speed
                    inference_time = speed_info.get('inference', 0.0) if speed_info else 0.0

                st.success(f"Analysis complete! Detected **{detected_count}** crack region(s) in {inference_time:.1f} ms.")

                st.markdown("### 📊 Side-by-Side Comparison")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📷 Original Image")
                    st.image(input_image, use_container_width=True, caption=f"Original: {uploaded_file.name}")

                with col2:
                    st.subheader("🎯 Segmented Cracks")
                    st.image(
                        annotated_rgb,
                        use_container_width=True,
                        caption=f"Segmentation Mask (conf ≥ {confidence_slider:.2f}, no labels)"
                    )
                    
                    result_pil = Image.fromarray(annotated_rgb)
                    buf = io.BytesIO()
                    result_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="📥 Download Segmented Image",
                        data=byte_im,
                        file_name=f"segmented_{uploaded_file.name}",
                        mime="image/png"
                    )

                if detected_count > 0 and masks is not None:
                    with st.expander("🔍 View Technical Detection Details", expanded=False):
                        st.write(f"- **Total Cracks Identified:** {detected_count}")
                        st.write(f"- **Inference Speed:** {inference_time:.2f} ms")
                        if hasattr(results[0], 'orig_shape'):
                            st.write(f"- **Original Resolution:** {results[0].orig_shape[1]}x{results[0].orig_shape[0]} px")
                        if boxes is not None and boxes.conf is not None:
                            conf_list = [f"{float(c):.3f}" for c in boxes.conf.cpu().numpy()]
                            st.write(f"- **Confidence Scores:** {', '.join(conf_list)}")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

else:
    st.markdown("---")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("#### 1. Upload Image")
        st.markdown("Select a photo of concrete, masonry, or asphalt containing cracks.")
    with col_info2:
        st.markdown("#### 2. Adjust Sensitivity")
        st.markdown("Use the confidence slider in the sidebar to fine-tune crack detection threshold (default 0.15).")
    with col_info3:
        st.markdown("#### 3. View Masks")
        st.markdown("Inspect segmented crack contours cleanly overlaid without distracting bounding box text.")