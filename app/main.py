import streamlit as st
import numpy as np
import pandas as pd
from detect import DetectCars, DetectEngine, Tracker, draw_bbox, get_processed_video
from pathlib import Path


engine = DetectEngine(endpoint_url=None, model_variant='m')

ASSESTS = Path(__file__).resolve().parent / 'assets'
demo_files = {i.name: i for i in list(ASSESTS.glob("*.mp4"))}


body = """
<h1> Robust Car Detection and Directional Analysis </h1> <br>
<h5> Lal Krishna Arjun K </h5>
<h5> AA.SC.P2MCA2207077 </h5>
<h5> Mini Project </h5>
<br>
"""
st.markdown(body=body, unsafe_allow_html=True)


upload_options = st.selectbox(
    'Choose video file...', ['-'] + list(demo_files.keys()) + ['Upload a File']
)
if upload_options == 'Upload a File':
    uploaded_file = st.file_uploader("Choose video file...")
    video_bytes = uploaded_file.read() if uploaded_file else None
elif upload_options == '-': video_bytes = False
else:
    video_bytes = demo_files[upload_options].read_bytes() 


if video_bytes and st.button("Start"):
    
    with st.status("Processing Video", expanded=True) as status:
        st.write("Extracting frames...")
        detection = DetectCars(video_bytes, engine=engine)
        st.write(f"{detection.is_cached=}")
        detection.get_frames()

        st.write("Detecting cars in frames...")
        detection.detect_cars_in_frames()

        # st.write("Saving processed frames...")
        # video_bytes = detection.get_processed_video()

        status.update(label="Processed", state="complete", expanded=False)
    


    with st.status("Analysing Video", expanded=True) as status:
        st.write("Tracking individual cars...")
        tracker = Tracker(hash=detection.video_hash, frame_rate=detection.fps)
        st.write("Analysing directions...")
        tracker.analyse(detection.frames)
        analysed_result = tracker.get_analysed_result()

        st.write("Drawing bboxes...")
        frames = draw_bbox(
            direction_result=tracker.direction_result,
            frames=detection.frames, result_df=tracker.result_df
        )
        st.write("Saving processed frames...")
        video_bytes = get_processed_video(
            frames=frames, fps=detection.fps, size=detection.size
        )
        
        status.update(label="Analysed", state="complete", expanded=False)

    col1, col2 = st.columns([2, 3])
    with col1:
   
        st.header("Video stats", divider="rainbow")
        st.table(analysed_result)

    with col2:
        st.header("Processed video", divider="rainbow")

        st.video(video_bytes)





