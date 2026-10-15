"""
AI-Based Real-Time Fire and Smoke Detection System.
Streamlit Web Application & Safety Monitoring Dashboard.
Final-Year Academic Computer Science & Engineering Project.
"""

from datetime import datetime
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Application package imports
from src.alert_manager import AlertManager
from src.confirmation import DetectionStatus, MultiFrameConfirmation
from src.database import AlertDatabase
from src.detector import FireSmokeDetector, ModelValidationError
from src.evaluation import EvaluationEngine
from src.utils import load_config, validate_model_classes


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Main Background & Text */
    .main {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Header Container */
    .system-header {
        background: linear-gradient(135deg, #1f1b24 0%, #291515 50%, #161a23 100%);
        padding: 18px 24px;
        border-radius: 10px;
        border: 1px solid #ff4b4b44;
        margin-bottom: 20px;
    }
    .system-header h1 {
        margin: 0;
        font-size: 1.9rem;
        color: #ff4b4b;
        letter-spacing: 1px;
    }
    .system-header p {
        margin: 4px 0 0 0;
        color: #b0b4be;
        font-size: 0.95rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #1a1e27;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-title {
        color: #8c93a0;
        font-size: 0.85rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    /* Status Badges */
    .badge-confirmed {
        background-color: #d32f2f;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-possible {
        background-color: #f57c00;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-safe {
        background-color: #388e3c;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Safety Disclaimer Box */
    .disclaimer-box {
        background-color: #1a1612;
        border: 1px solid #ff980055;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #ffb74d;
        margin-top: 25px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# State & Resource Management
# -----------------------------------------------------------------------------
@st.cache_resource
def get_config():
    return load_config("config/config.yaml")


@st.cache_resource
def get_database(db_path: str):
    return AlertDatabase(db_path=db_path)


def get_detector_instance(model_path: str, conf_threshold: float, iou_threshold: float):
    """Safely initialize detector or return None with error explanation."""
    try:
        detector = FireSmokeDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        return detector, None
    except ModelValidationError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error loading detector: {e}"


# -----------------------------------------------------------------------------
# Sidebar Navigation & Settings
# -----------------------------------------------------------------------------
config = get_config()
db = get_database(config["database"]["db_path"])

st.sidebar.image(
    "https://img.icons8.com/fluency/96/fire-element.png",
    width=64,
)
st.sidebar.title("Fire & Smoke AI")
st.sidebar.caption("Computer Vision Safety System")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "📹 Live Camera Monitoring",
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📜 Alert History",
        "📈 Model Evaluation",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Configuration")

# Model Path Setting
configured_model_path = st.sidebar.text_input(
    "Model Weights Path",
    value=config["model"]["path"],
    help="Path to trained YOLO weights file (e.g. models/fire_smoke_yolov8n.pt)",
)

# Confidence & IOU Sliders
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(config["model"]["default_confidence"]),
    step=0.05,
    help="Detections below this confidence will be ignored.",
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold (NMS)",
    min_value=0.20,
    max_value=0.80,
    value=float(config["model"]["iou_threshold"]),
    step=0.05,
    help="Non-Maximum Suppression overlap threshold.",
)

with st.sidebar.expander("⏱️ Temporal Confirmation Settings", expanded=False):
    window_size = st.number_input(
        "Window Size (Frames)",
        min_value=2,
        max_value=30,
        value=int(config["confirmation"]["window_size"]),
    )
    fire_req = st.number_input(
        "Fire Required Frames",
        min_value=1,
        max_value=window_size,
        value=min(int(config["confirmation"]["fire_required_frames"]), window_size),
    )
    smoke_req = st.number_input(
        "Smoke Required Frames",
        min_value=1,
        max_value=window_size,
        value=min(int(config["confirmation"]["smoke_required_frames"]), window_size),
    )
    alert_cooldown = st.number_input(
        "Alert Cooldown (Seconds)",
        min_value=1,
        max_value=120,
        value=int(config["confirmation"]["alert_cooldown_seconds"]),
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Final-Year Project: CSE\n"
    "Based on IEEE Access 2025 Fire Detection Network baseline research."
)


# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="system-header">
        <h1>AI FIRE & SMOKE DETECTION SYSTEM</h1>
        <p>Real-Time Computer Vision Safety Monitoring & Temporal Alert Confirmation</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD
# -----------------------------------------------------------------------------
if nav_selection == "📊 Dashboard":
    st.subheader("System Overview & Live Health Status")

    # Probe Model Status
    detector, model_err = get_detector_instance(configured_model_path, conf_threshold, iou_threshold)
    counts = db.get_alert_counts()

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #2196F3;">
                <div class="metric-title">Total Alerts Recorded</div>
                <div class="metric-value">{counts['total']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #FF5252;">
                <div class="metric-title">Confirmed Fire Alerts</div>
                <div class="metric-value">{counts['fire']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #B0BEC5;">
                <div class="metric-title">Confirmed Smoke Alerts</div>
                <div class="metric-value">{counts['smoke']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left-color: #4CAF50;">
                <div class="metric-title">Confirmation Ratio</div>
                <div class="metric-value">{fire_req}/{window_size} <span style="font-size: 1rem; color: #888;">frames</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # System Status Diagnostics
    st.markdown("#### Diagnostic Status")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("**AI Model Status:**")
        if detector is not None and detector.is_loaded:
            st.success(f"✅ Verified Model Active ({len(detector.class_names)} classes: {detector.class_names})")
        else:
            st.error(f"❌ Model Unavailable: {model_err}")
            st.info(
                "💡 **How to configure a model:**\n"
                "- Train a custom model using: `python scripts/train.py`\n"
                "- Or place custom weights in `models/fire_smoke_yolov8n.pt`\n"
                "- Generic COCO models cannot detect Fire/Smoke without training."
            )

    with d2:
        st.markdown("**Database & Storage:**")
        if Path(config["database"]["db_path"]).is_file():
            st.success(f"✅ SQLite Database Online (`{config['database']['db_path']}`)")
        else:
            st.info(f"ℹ️ SQLite Database will initialize upon first alert (`{config['database']['db_path']}`)")

    st.markdown("---")
    st.markdown("#### Recent Confirmed Safety Alerts")
    recent_alerts = db.get_alerts(limit=10)
    if recent_alerts:
        df_recent = pd.DataFrame(recent_alerts)
        st.dataframe(
            df_recent[["id", "timestamp", "detection_type", "confidence", "source", "status", "notes"]],
            use_container_width=True,
        )
    else:
        st.info("No safety alerts recorded yet. System state is clean.")

    # Base Paper Citation & Research Note
    with st.expander("📚 Academic Citation & Research Foundation"):
        st.markdown(
            """
            **Research Foundation:**
            * D. H. Shin, J. M. Kang, and T. Cheong, *"FDN: A Real-Time Ensemble Fire Detection Network,"* **IEEE Access**, 2025. DOI: 10.1109/ACCESS.2025.3575057.
            
            **Student Project Architecture Note:**
            This project implements an accessible, high-performance YOLO baseline coupled with an OpenCV temporal confirmation filter to achieve false alarm reduction suitable for real-time edge processing on laptops.
            """
        )

    # Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Safety Prototype Disclaimer:</strong> This software is an academic engineering demonstration. 
            AI-based vision models are subject to environmental factors (camera resolution, lighting, reflective surfaces, smoke-like dust). 
            This system must NOT replace certified industrial fire alarm, smoke sensor, or sprinkler safety systems.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 2: LIVE CAMERA MONITORING
# -----------------------------------------------------------------------------
elif nav_selection == "📹 Live Camera Monitoring":
    st.subheader("Real-Time Webcam / CCTV Video Stream")

    detector, model_err = get_detector_instance(configured_model_path, conf_threshold, iou_threshold)
    if detector is None:
        st.error(f"Cannot start live monitoring: {model_err}")
        st.stop()

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
    with col_ctrl1:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    with col_ctrl2:
        start_btn = st.button("▶️ Start Monitoring", use_container_width=True, type="primary")
    with col_ctrl3:
        stop_btn = st.button("⏹️ Stop Monitoring", use_container_width=True)

    # State tracking for camera
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False

    # Layout for video and status panels
    col_video, col_status = st.columns([3, 2])
    with col_video:
        video_placeholder = st.empty()
    with col_status:
        status_placeholder = st.empty()
        alert_placeholder = st.empty()

    if st.session_state.camera_running:
        # Initialize confirmation & alert managers
        confirmation = MultiFrameConfirmation(
            window_size=window_size,
            fire_required_frames=fire_req,
            smoke_required_frames=smoke_req,
        )
        alert_mgr = AlertManager(db=db, cooldown_seconds=alert_cooldown)

        # Attempt OpenCV Capture
        # On Windows, cv2.CAP_DSHOW prevents slow startup
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)

        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(int(camera_index))

        if not cap.isOpened():
            st.error(
                f"❌ Failed to access camera at index {camera_index}.\n\n"
                "**Troubleshooting Checklist:**\n"
                "- Check if another application (Zoom, Teams, Camera App) is currently using the camera.\n"
                "- Verify Windows Camera Privacy Settings (Settings -> Privacy & Security -> Camera).\n"
                "- Try camera index 1 or 2 if external USB webcam is plugged in."
            )
            st.session_state.camera_running = False
        else:
            try:
                frame_count = 0
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame from camera stream.")
                        break

                    frame_count += 1

                    # Run Detection
                    detections = detector.detect(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                    annotated_bgr = detector.annotate_frame(frame, detections)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    # Update Confirmation State
                    c_state = confirmation.update(detections)
                    new_alerts = alert_mgr.process_confirmation(c_state, source=f"Webcam Index {camera_index}")

                    # Display Live Frame
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

                    # Update Status UI
                    fire_info = c_state["fire"]
                    smoke_info = c_state["smoke"]

                    fire_badge = (
                        '<span class="badge-confirmed">CONFIRMED FIRE</span>'
                        if fire_info["is_confirmed"]
                        else (
                            '<span class="badge-possible">POSSIBLE FIRE</span>'
                            if fire_info["status"] == DetectionStatus.POSSIBLE
                            else '<span class="badge-safe">NO FIRE</span>'
                        )
                    )

                    smoke_badge = (
                        '<span class="badge-confirmed">CONFIRMED SMOKE</span>'
                        if smoke_info["is_confirmed"]
                        else (
                            '<span class="badge-possible">POSSIBLE SMOKE</span>'
                            if smoke_info["status"] == DetectionStatus.POSSIBLE
                            else '<span class="badge-safe">NO SMOKE</span>'
                        )
                    )

                    status_html = f"""
                    <div style="background: #1a1e27; padding: 14px; border-radius: 8px; margin-bottom: 12px;">
                        <h4 style="margin: 0 0 10px 0; color: #fff;">Temporal Confirmation Monitor</h4>
                        <div style="margin-bottom: 8px;">
                            <strong>Fire Status:</strong> {fire_badge}<br>
                            <small style="color: #aaa;">Detections in window: {fire_info['count']}/{fire_info['window_size']} (Req: {fire_req})</small>
                        </div>
                        <div>
                            <strong>Smoke Status:</strong> {smoke_badge}<br>
                            <small style="color: #aaa;">Detections in window: {smoke_info['count']}/{smoke_info['window_size']} (Req: {smoke_req})</small>
                        </div>
                    </div>
                    """
                    status_placeholder.markdown(status_html, unsafe_allow_html=True)

                    if new_alerts:
                        for al in new_alerts:
                            alert_placeholder.error(
                                f"🚨 **ALERT RECORDED [{al['timestamp']}]:** {al['detection_type']} "
                                f"with {al['confidence']*100:.1f}% confidence logged to database!"
                            )

                    # Brief pause for cooperative thread yielding
                    time.sleep(0.01)

            finally:
                cap.release()
                video_placeholder.empty()
    else:
        video_placeholder.info("Camera is currently stopped. Click 'Start Monitoring' to begin live stream.")


# -----------------------------------------------------------------------------
# PAGE 3: IMAGE DETECTION
# -----------------------------------------------------------------------------
elif nav_selection == "🖼️ Image Detection":
    st.subheader("Single Image Fire & Smoke Detection")
    st.caption("Upload a JPG, JPEG, or PNG image to inspect detection bounding boxes and confidence scores.")

    detector, model_err = get_detector_instance(configured_model_path, conf_threshold, iou_threshold)
    if detector is None:
        st.error(f"Cannot run detection: {model_err}")
        st.stop()

    uploaded_file = st.file_uploader("Select Image File", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is not None:
        # Load image with PIL and convert to OpenCV BGR
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_image)
        bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run inference
        t0 = time.time()
        detections = detector.detect(bgr_frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        infer_time_ms = (time.time() - t0) * 1000.0

        # Annotate
        annotated_bgr = detector.annotate_frame(bgr_frame, detections)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown("**Original Image:**")
            st.image(pil_image, use_container_width=True)
        with col_img2:
            st.markdown(f"**AI Detection Result (Inference: {infer_time_ms:.1f} ms):**")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")
        st.markdown(f"#### Summary: `{detector.format_summary(detections)}`")

        if detections:
            det_rows = []
            for i, d in enumerate(detections, 1):
                x1, y1, x2, y2 = d["box"]
                det_rows.append({
                    "#": i,
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']*100:.1f}%",
                    "Bounding Box (X1, Y1, X2, Y2)": f"[{x1}, {y1}, {x2}, {y2}]",
                    "Box Area (Pixels)": (x2 - x1) * (y2 - y1),
                })
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
        else:
            st.success("✅ Clean Scene: No fire or smoke detected above the confidence threshold.")


# -----------------------------------------------------------------------------
# PAGE 4: VIDEO DETECTION
# -----------------------------------------------------------------------------
elif nav_selection == "🎥 Video Detection":
    st.subheader("Video File Detection & Temporal Confirmation")
    st.caption("Upload a recorded video (MP4, AVI, MOV) to process frame-by-frame and log confirmed alerts.")

    detector, model_err = get_detector_instance(configured_model_path, conf_threshold, iou_threshold)
    if detector is None:
        st.error(f"Cannot run detection: {model_err}")
        st.stop()

    uploaded_video = st.file_uploader("Select Video File", type=["mp4", "avi", "mov", "mkv"])

    col_vopt1, col_vopt2 = st.columns(2)
    with col_vopt1:
        frame_skip = st.select_slider(
            "Frame Skip (Speed Optimizer)",
            options=[1, 2, 3, 5],
            value=1,
            help="Process every Nth frame (1 = every frame; 2 = every 2nd frame, 2x faster).",
        )
    with col_vopt2:
        export_annotated = st.checkbox("Generate Annotated Output Video", value=True)

    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix)
        tfile.write(uploaded_video.read())
        tfile.close()
        video_path = tfile.name

        start_process = st.button("🚀 Process Video", type="primary")

        if start_process:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_placeholder = st.empty()

            output_temp_path = None
            writer = None
            if export_annotated:
                output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                output_temp.close()
                output_temp_path = output_temp.name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_temp_path, fourcc, fps / frame_skip, (width, height))

            confirmation = MultiFrameConfirmation(
                window_size=window_size,
                fire_required_frames=fire_req,
                smoke_required_frames=smoke_req,
            )
            alert_mgr = AlertManager(db=db, cooldown_seconds=alert_cooldown)

            all_video_alerts = []
            frame_idx = 0
            processed_count = 0
            start_time = time.time()

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_skip > 1 and (frame_idx % frame_skip != 0):
                        continue

                    processed_count += 1
                    detections = detector.detect(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                    annotated_bgr = detector.annotate_frame(frame, detections)

                    c_state = confirmation.update(detections)
                    new_alerts = alert_mgr.process_confirmation(
                        c_state, source=f"Uploaded Video ({uploaded_video.name})"
                    )
                    if new_alerts:
                        all_video_alerts.extend(new_alerts)

                    if writer is not None:
                        writer.write(annotated_bgr)

                    # Update preview every few frames
                    if processed_count % 5 == 0:
                        preview_placeholder.image(
                            cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                            caption=f"Processing Frame {frame_idx}/{total_frames}",
                            channels="RGB",
                            use_container_width=True,
                        )
                        pct = min(1.0, frame_idx / max(total_frames, 1))
                        progress_bar.progress(pct)
                        status_text.text(f"Processed {processed_count} frames... ({int(pct*100)}%)")

            finally:
                cap.release()
                if writer is not None:
                    writer.release()
                # Clean up uploaded raw file
                try:
                    os.unlink(video_path)
                except Exception:
                    pass

            elapsed = round(time.time() - start_time, 2)
            progress_bar.progress(1.0)
            status_text.success(f"Processing finished in {elapsed}s ({processed_count} frames processed).")

            st.markdown(f"**Alerts Triggered During Video:** {len(all_video_alerts)}")
            if all_video_alerts:
                st.dataframe(pd.DataFrame(all_video_alerts)[["timestamp", "detection_type", "confidence", "message"]])
            else:
                st.info("No confirmed fire or smoke events occurred in this video.")

            if output_temp_path and Path(output_temp_path).is_file():
                with open(output_temp_path, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=vf.read(),
                        file_name=f"annotated_{uploaded_video.name}",
                        mime="video/mp4",
                    )


# -----------------------------------------------------------------------------
# PAGE 5: ALERT HISTORY
# -----------------------------------------------------------------------------
elif nav_selection == "📜 Alert History":
    st.subheader("Confirmed Safety Alert Log")
    st.caption("Persistent record of all confirmed fire and smoke events stored in SQLite.")

    col_filter, col_export, col_clear = st.columns([2, 1, 1])
    with col_filter:
        filter_type = st.selectbox("Filter by Detection Type", ["ALL", "FIRE", "SMOKE"])
    with col_export:
        export_btn = st.button("📥 Export to CSV", use_container_width=True)
    with col_clear:
        confirm_clear = st.checkbox("Confirm Clear Database")
        clear_btn = st.button("🗑️ Clear History", disabled=not confirm_clear, use_container_width=True)

    if clear_btn and confirm_clear:
        deleted = db.clear_alerts()
        st.success(f"Successfully purged {deleted} alert records.")

    if export_btn:
        csv_path, row_count = db.export_alerts_csv()
        st.success(f"Exported {row_count} alerts to `{csv_path}`.")
        with open(csv_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="Click to Download CSV File",
                data=f.read(),
                file_name=Path(csv_path).name,
                mime="text/csv",
            )

    alerts = db.get_alerts(detection_type=filter_type, limit=500)
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        st.dataframe(
            df_alerts[["id", "timestamp", "detection_type", "confidence", "source", "status", "notes"]],
            use_container_width=True,
            height=450,
        )
    else:
        st.info("No alerts found matching the current filter.")


# -----------------------------------------------------------------------------
# PAGE 6: MODEL EVALUATION
# -----------------------------------------------------------------------------
elif nav_selection == "📈 Model Evaluation":
    st.subheader("Model Evaluation & Hardware Benchmark")
    st.caption(
        "Measure genuine object detection performance (mAP, Precision, Recall, F1) on labeled test data, "
        "and test real-time inference latency."
    )

    evaluator = EvaluationEngine(model_path=configured_model_path)

    tab_eval, tab_bench, tab_academic = st.tabs([
        "📊 Test Dataset Evaluation",
        "⚡ Latency & FPS Benchmark",
        "📖 Academic Research Alignment",
    ])

    with tab_eval:
        st.markdown("#### Evaluate on Labeled Dataset")
        st.markdown(
            "Ultralytics YOLO computes standard **mAP@0.5** and **mAP@0.5:0.95** "
            "by comparing predicted bounding boxes against ground-truth YOLO text labels in `dataset/labels/test`."
        )

        dataset_yaml_path = st.text_input("Dataset YAML Path", value="config/dataset.yaml")
        split_to_eval = st.selectbox("Split", ["test", "val"], index=0)

        eval_btn = st.button("Run Full Evaluation", type="primary")

        if eval_btn:
            with st.spinner("Running YOLO validation pipeline..."):
                try:
                    res = evaluator.evaluate_on_dataset(dataset_yaml=dataset_yaml_path, split=split_to_eval)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Precision", f"{res['precision']:.4f}")
                    m2.metric("Recall", f"{res['recall']:.4f}")
                    m3.metric("F1-Score", f"{res['f1_score']:.4f}")
                    m4.metric("mAP@0.5", f"{res['mAP50']:.4f}")

                    chart_path = evaluator.generate_metrics_chart(res)
                    if Path(chart_path).is_file():
                        st.image(chart_path, caption="Evaluation Metrics Summary", use_container_width=True)

                    if res.get("per_class"):
                        st.markdown("**Per-Class Breakdown:**")
                        st.dataframe(pd.DataFrame(res["per_class"]))

                except FileNotFoundError as e:
                    st.warning(f"Evaluation cannot run: {e}")
                    st.info(
                        "📌 **Why are metrics not displayed yet?**\n"
                        "Object detection metrics (mAP, Precision, Recall) require a genuine labeled test set "
                        "with ground-truth bounding boxes.\n\n"
                        "To prepare the test set:\n"
                        "1. Download a verified dataset (e.g. Roboflow Fire & Smoke YOLOv8 format).\n"
                        "2. Place images and labels in `dataset/images/test` and `dataset/labels/test`.\n"
                        "3. Click 'Run Full Evaluation' to compute genuine metrics."
                    )

    with tab_bench:
        st.markdown("#### Hardware Inference Benchmark")
        st.markdown(
            "Measures actual frame processing time and real-time frames-per-second (FPS) on your CPU/GPU."
        )

        num_iterations = st.slider("Iterations to Average", min_value=10, max_value=100, value=30)
        run_bench = st.button("⚡ Run Hardware Benchmark")

        if run_bench:
            with st.spinner("Benchmarking inference speed..."):
                try:
                    b_res = evaluator.benchmark_inference_speed(num_iterations=num_iterations)
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Average Latency", f"{b_res['mean_latency_ms']} ms")
                    b2.metric("Throughput (FPS)", f"{b_res['fps']} FPS")
                    b3.metric("Min / Max Latency", f"{b_res['min_latency_ms']} / {b_res['max_latency_ms']} ms")

                    if b_res["fps"] >= 15.0:
                        st.success(f"🚀 Real-Time Viable: Model achieves {b_res['fps']} FPS on current hardware.")
                    else:
                        st.warning(
                            f"⚠️ Current throughput ({b_res['fps']} FPS) is lower than 15 FPS. "
                            "Consider enabling frame skipping (skip=2) or using GPU."
                        )
                except Exception as e:
                    st.error(f"Benchmark error: {e}")

    with tab_academic:
        st.markdown(
            """
            #### Base Paper Analysis
            * **Reference**: D. H. Shin, J. M. Kang, and T. Cheong, *"FDN: A Real-Time Ensemble Fire Detection Network,"* IEEE Access, vol. 13, 2025.
            * **Motivation**: The base paper identifies that real-time fire and smoke detection systems often suffer from high false alarm rates caused by ambient light reflections and fast transient smoke changes.
            * **Our CSE Project Contribution**:
              1. **Modular YOLO Pipeline**: Lightweight YOLO architecture suitable for standard laptop CPU execution.
              2. **Temporal Multi-Frame Confirmation Filter**: Instead of triggering alerts on single-frame optical glitches, our system uses a configurable rolling temporal window (e.g., 3/5 frames).
              3. **Alert Cooldown & Persistent SQLite Database**: Prevents alarm flood fatigue while keeping auditable records.
              4. **Interactive Safety Monitoring Dashboard**: Complete Streamlit application supporting live webcam, uploaded video, and image inspection.
            """
        )
