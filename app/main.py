import streamlit as st
import numpy as np
import pandas as pd
from detect import DetectCars



# st.set_page_config(layout="wide")

uploaded_file = st.file_uploader("Choose video file...")




if uploaded_file:
    
    with st.status("Analysing Video", expanded=True) as status:
        st.write("Extracting frames...")
        detection = DetectCars(uploaded_file)
        st.write(f"{detection.video_hash=} {detection.is_cached=}")
        detection.get_frames()
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





