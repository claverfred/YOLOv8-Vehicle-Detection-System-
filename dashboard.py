import time
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    last_seen_frame: int
    counted_up: bool = False
    counted_down: bool = False


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _normalize_label(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"cars", "car"}:
        return "car"
    if n in {"motorbike", "motorbike ", "motorcycle", "bike"}:
        return "motorbike"
    return n


def _counts_from_results(results, class_names: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    boxes = results[0].boxes
    if boxes is None or boxes.cls is None:
        return counts
    for cls in boxes.cls.tolist():
        name = _normalize_label(class_names.get(int(cls), str(int(cls))))
        counts[name] = counts.get(name, 0) + 1
    return counts


def _centroids_from_results(results) -> list[tuple[int, int]]:
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    centroids: list[tuple[int, int]] = []
    for x1, y1, x2, y2 in xyxy:
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        centroids.append((cx, cy))
    return centroids


def _match_tracks(
    tracks: dict[int, Track],
    detections: list[tuple[int, int]],
    frame_idx: int,
    max_distance_px: int,
    max_age_frames: int,
    next_id: int,
) -> tuple[dict[int, Track], int]:
    # Drop stale tracks
    tracks = {
        tid: t
        for tid, t in tracks.items()
        if (frame_idx - t.last_seen_frame) <= max_age_frames
    }

    if not detections:
        return tracks, next_id

    used_det = set()

    # Greedy nearest-neighbor matching
    for tid, t in list(tracks.items()):
        best_j = None
        best_d = None
        for j, det in enumerate(detections):
            if j in used_det:
                continue
            d = int(np.hypot(det[0] - t.centroid[0], det[1] - t.centroid[1]))
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d is not None and best_d <= max_distance_px:
            det = detections[best_j]
            used_det.add(best_j)
            tracks[tid] = Track(
                track_id=tid,
                centroid=det,
                last_seen_frame=frame_idx,
                counted_up=t.counted_up,
                counted_down=t.counted_down,
            )

    # Create new tracks for unmatched detections
    for j, det in enumerate(detections):
        if j in used_det:
            continue
        tracks[next_id] = Track(track_id=next_id, centroid=det, last_seen_frame=frame_idx)
        next_id += 1

    return tracks, next_id


def _planning_insights(car_count: int, bike_count: int, density_per_m2: float) -> list[str]:
    total = car_count + bike_count
    share_bike = _safe_div(bike_count, total)

    insights: list[str] = []

    if total == 0:
        return [
            "No vehicles detected in this scene. Try a clearer street view image/video or lower the confidence threshold.",
        ]


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Urban Mobility Planning", page_icon="🗺️", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown(
    """ <style>
    .stApp { background-color: #0E1117; color: white; }
    h1, h2, h3 { color: #00FFAA; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Urban Mobility Planning Dashboard")
st.markdown("### Using **car & motorbike detections** for Traffic counting")


# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35)

st.sidebar.subheader("Model")
model_source = st.sidebar.radio("Weights source", ["Use local path", "Upload .pt weights"], index=0)

weights_path = None
if model_source == "Use local path":
    weights_path = st.sidebar.text_input("Path to weights (.pt)", value="best.pt")
else:
    uploaded_weights = st.sidebar.file_uploader("Upload YOLO weights", type=["pt"])
    if uploaded_weights is not None:
        tweights = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tweights.write(uploaded_weights.read())
        weights_path = tweights.name

mode = st.sidebar.radio("Select Mode", ["Image (snapshot)", "Video (traffic flow)"])


# ---------------- LOAD MODEL ----------------
if not weights_path:
    st.stop()

try:
    model = YOLO(weights_path)
except Exception as e:
    st.error(f"Could not load YOLO weights from `{weights_path}`.\n\nError: {e}")
    st.stop()


def _image_area_m2(width_px: int, height_px: int, m_per_px: float | None) -> float:
    if not m_per_px:
        return 0.0
    return float(width_px * height_px) * float(m_per_px**2)



# ---------------- IMAGE MODE ----------------
if mode == "Image (snapshot)":
    uploaded_file = st.file_uploader("📤 Upload street image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file).convert("RGB")
        col1.image(image, caption="Original", use_container_width=True)

        results = model(image, conf=confidence)
        annotated = results[0].plot()  # BGR numpy array (OpenCV order)
        col2.image(annotated, caption="Detections", channels="BGR", use_container_width=True)

        class_names = model.names if hasattr(model, "names") else {}
        counts = _counts_from_results(results, class_names)

        car_count = counts.get("car", 0)
        motorbike_count = counts.get("motorbike", 0)
        total = car_count + motorbike_count

        img_w, img_h = image.size
        #area_m2 = _image_area_m2(img_w, img_h, meters_per_pixel if use_calibration else None)
        #density = _safe_div(total, area_m2) if area_m2 else 0.0

        st.markdown("---")

# ---------------- COUNT OBJECTS ----------------
        boxes = results[0].boxes
        class_names = model.names
        counts = {}

        if boxes is not None:
            for cls in boxes.cls:
                name = class_names[int(cls)]
                counts[name] = counts.get(name, 0) + 1
    
        st.subheader("📊 Detection Summary")
        
        car_count = counts.get("car", 0) + counts.get("cars", 0)
        motorbike_count = counts.get("motorbike", 0) + counts.get("motorcycle", 0)
        total_objects = sum(counts.values())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div style="
                    background: rgba(0,255,170,0.08);
                    padding: 12px;
                    border-radius: 12px;
                    text-align: center;
                    border: 1px solid rgba(0,255,170,0.3);
                ">
                    <h4 style="color:#00FFAA; margin-bottom:5px;">🚗 Cars</h4>
                    <h2 style="color:white; margin:0;">{car_count}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="
                    background: rgba(0,150,255,0.08);
                    padding: 12px;
                    border-radius: 12px;
                    text-align: center;
                    border: 1px solid rgba(0,150,255,0.3);
                ">
                    <h4 style="color:#4DA3FF; margin-bottom:5px;">🏍 Motorbikes</h4>
                    <h2 style="color:white; margin:0;">{motorbike_count}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    padding: 12px;
                    border-radius: 12px;
                    text-align: center;
                    border: 1px solid rgba(255,255,255,0.2);
                ">
                    <h4 style="color:white; margin-bottom:5px;">📦 Total Objects</h4>
                    <h2 style="color:#00FFAA; margin:0;">{total_objects}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Mode share
        st.markdown("#### Mode share")
        if total > 0:
            share_df = pd.DataFrame(
                {
                    "mode": ["car", "motorbike"],
                    "count": [car_count, motorbike_count],
                }
            )
            st.bar_chart(share_df.set_index("mode"))
        else:
            st.info("No vehicles detected — try another image or lower confidence.")


        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download annotated image",
            data=cv2.imencode(".jpg", annotated)[1].tobytes(),
            file_name="urban_mobility_detections.jpg",
            mime="image/jpeg",
        )


# ---------------- VIDEO MODE ----------------
if mode == "Video (traffic flow)":
    video_file = st.file_uploader("📤 Upload traffic video", type=["mp4", "mov", "avi"])

    st.sidebar.subheader("Traffic counting")
    count_line_y_pct = st.sidebar.slider("Counting line (height %)", 5, 95, 60)
    frame_stride = st.sidebar.slider("Increase or Decrease speed", 1, 10, 2)
    max_distance_px = st.sidebar.slider("Tracking match distance (px)", 10, 200, 60)

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        stframe = st.empty()
        chart_area = st.empty()

        # st.warning("Processing video... (tip: increase 'Process every N frames' for speed)")

        tracks: dict[int, Track] = {}
        next_id = 1
        max_age_frames = int(fps * 2)  # keep tracks ~2s without updates

        up_count = 0
        down_count = 0
        per_bin: list[dict] = []

        frame_idx = 0
        start_t = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_stride != 0:
                continue

            h, w = frame.shape[:2]
            line_y = int(h * (count_line_y_pct / 100.0))

            # Ultralytics expects numpy frames in BGR order, which is what OpenCV reads.
            results = model(frame, conf=confidence)

            annotated_frame = results[0].plot()  # BGR

            # Count crossings using centroid tracking (all detected objects, demo-level)
            centroids = _centroids_from_results(results)
            prev_centroids = {tid: t.centroid for tid, t in tracks.items()}

            tracks, next_id = _match_tracks(
                tracks=tracks,
                detections=centroids,
                frame_idx=frame_idx,
                max_distance_px=max_distance_px,
                max_age_frames=max_age_frames,
                next_id=next_id,
            )

            # Crossing logic: compare previous vs current y relative to line
            for tid, t in list(tracks.items()):
                prev = prev_centroids.get(tid)
                if prev is None:
                    continue
                prev_y = prev[1]
                curr_y = t.centroid[1]

                if prev_y < line_y <= curr_y and not t.counted_down:
                    down_count += 1
                    tracks[tid].counted_down = True
                if prev_y > line_y >= curr_y and not t.counted_up:
                    up_count += 1
                    tracks[tid].counted_up = True

            # Draw counting line
            cv2.line(annotated_frame, (0, line_y), (w, line_y), (170, 255, 0), 2)  # #00FFAA in BGR
            cv2.putText(
                annotated_frame,
                f"Up: {up_count}  Down: {down_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            # Build a simple per-10s flow series
            elapsed = time.time() - start_t
            bin_s = int(elapsed // 10) * 10
            per_bin.append({"t_bin_s": bin_s, "up": up_count, "down": down_count})

            if len(per_bin) >= 5:
                df = pd.DataFrame(per_bin).groupby("t_bin_s", as_index=False).max()
                df["total"] = df["up"] + df["down"]
                chart_area.line_chart(df.set_index("t_bin_s")[["up", "down", "total"]])

        cap.release()
        st.success("Video processing completed!")

        st.markdown("---")
        st.subheader("🧭 Planning interpretation (video)")
        st.write(
            f"Estimated corridor flow (demo): **Up={up_count}**, **Down={down_count}**, **Total crossings={up_count + down_count}**."
        )
       


st.markdown("---")
st.info("Urban Mobility Planning | Developed by Allen, Cleverfred and Frank")