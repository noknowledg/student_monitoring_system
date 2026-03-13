import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import os

# Load YOLOv8 model
model = YOLO(r"D:\New folder (4)\Classroom_best.pt")  # Make sure path is correct

st.title("🎥 Classroom Attention Detection")
st.write("Upload a video to check predictions using YOLOv8")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)  # Show uploaded video

    if st.button("Run Prediction"):
        cap = cv2.VideoCapture(video_path)

        # Video writer setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 10  # fallback FPS
        output_video_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        attentive_count = 0
        not_attentive_count = 0
        total_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Optional: downscale frame to save memory
            # frame = cv2.resize(frame, (640, 480))

            # Run YOLO inference on the frame
            results = model(frame)

            # Count instances
            for r in results:
                if r.boxes is not None:
                    classes = r.boxes.cls.cpu().numpy()
                    for c in classes:
                        if c == 0:  # Class 0 = attentive
                            attentive_count += 1
                        elif c == 1:  # Class 1 = not attentive
                            not_attentive_count += 1

            # Annotate frame
            annotated_frame = results[0].plot()
            out.write(annotated_frame)  # Write frame directly to video

            total_frames += 1
            # Optional: limit frames for testing
            # if total_frames >= 100:
            #     break

        cap.release()
        out.release()

        # Calculate percentages
        total_students = attentive_count + not_attentive_count
        if total_students > 0:
            attentive_percent = (attentive_count / total_students) * 100
            not_attentive_percent = (not_attentive_count / total_students) * 100
        else:
            attentive_percent = 0
            not_attentive_percent = 0

        # Display percentages
        st.success("✅ Prediction Completed!")
        st.write(f"🟢 Attentive Students: {attentive_percent:.2f}%")
        st.write(f"🔴 Not Attentive Students: {not_attentive_percent:.2f}%")

        # Show output video
        if os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.warning("⚠️ Could not find output video. Check YOLO save path.")
