import streamlit as st
import numpy as np
import pandas as pd
from detect import DetectCars, DetectEngine
from pathlib import Path


engine = DetectEngine(endpoint_url=None, model_variant='m')

ASSESTS = Path(__file__).resolve().parent / 'assets'
demo_files = {i.name: i for i in list(ASSESTS.glob("*.mp4"))}


body = """
<h1> Mini Project </h1> <br>
<h5> Lal Krishna Arjun K </h5>
<h5> AA.SC.P2MCA2207077 </h5>
<h5> Topic:  Robust Car Detection and Directional Analysis in Noisy Video Environments</h5>
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


if video_bytes:
    
    with st.status("Analysing Video", expanded=True) as status:
        st.write("Extracting frames...")
        detection = DetectCars(video_bytes, engine=engine)
        st.write(f"{detection.is_cached=}")
        detection.get_frames()

        st.write("Detecting cars in frames...")
        detection.detect_cars_in_frames()

        st.write("Saving processed frames...")
        video_bytes = detection.get_processed_video()

        status.update(label="Processed", state="complete", expanded=False)
    


    col1, col2 = st.columns([2, 3])
    with col1:
        st.header("Video stats", divider="rainbow")

        df = pd.DataFrame.from_dict({
            "index": [
                "Number of cars", 
                "Moving on left direction", 
                "Moving on right direction",
                "Moving on upward direction",
                "Moving on downward direction"
            ],
            "values": [3, 4, 2, 2, 1]
        })
        df = df.set_index('index')
        st.table(df)

    with col2:
        st.header("Processed video", divider="rainbow")

        st.video(video_bytes)





