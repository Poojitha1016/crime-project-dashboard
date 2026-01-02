import streamlit as st

import pandas as pd

import numpy as np

import tensorflow as tf

import pickle

import os

import cv2

from ultralytics import YOLO



# --- PAGE CONFIGURATION ---

# We've added a page_icon and set the theme to "dark"

st.set_page_config(

    page_title="Crime Analysis Dashboard",

    page_icon="🚨",

    layout="wide",

    initial_sidebar_state="expanded",

)



# --- MODEL AND ASSET LOADING ---

@st.cache_resource

def load_models():

    """Load all machine learning models and necessary assets."""

    # FFN Model

    ffn_model = tf.keras.models.load_model('./models/ffn_crime_classifier.h5')

    with open('./models/ffn_scaler.pkl', 'rb') as f:

        ffn_scaler = pickle.load(f)

    with open('./models/ffn_label_encoder.pkl', 'rb') as f:

        ffn_encoder = pickle.load(f)



    # LSTM Model

    lstm_model = tf.keras.models.load_model('./models/lstm_forecaster.h5')

    with open('./models/lstm_scaler.pkl', 'rb') as f:

        lstm_scaler = pickle.load(f)



    # YOLO Model

    yolo_model = YOLO('yolov8n.pt')



    return ffn_model, ffn_scaler, ffn_encoder, lstm_model, lstm_scaler, yolo_model



@st.cache_data

def load_data():

    """Load and preprocess the main CSV data."""

    df = pd.read_csv('./data/chicago_crime_data.csv', on_bad_lines='skip')

    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

    df.dropna(subset=['Date'], inplace=True) # Drop rows where date conversion failed

    

    # Pre-calculate features for charts

    df['Hour'] = df['Date'].dt.hour

    df['DayOfWeek'] = df['Date'].dt.day_name()

    return df



# Load all assets

ffn_model, ffn_scaler, ffn_encoder, lstm_model, lstm_scaler, yolo_model = load_models()

df = load_data()





# --- SIDEBAR NAVIGATION ---

st.sidebar.title("Navigation")

page = st.sidebar.radio("Choose a page:", ["Home", "Data Overview", "Crime Heatmap", "Crime Type Prediction", "Crime Count Forecast", "Video Object Detection"])



# --- PAGE 1: HOME ---

if page == "Home":

    st.title("Crime Prediction and Detection Dashboard 🚨")



    # --- NEW/UPDATED ---

    # 1. Use a colored "info" box for the welcome text.

    st.info("""

    **Welcome to the Crime Analysis Dashboard!** This project uses historical data and real-time video processing

    to provide insights into crime patterns in Chicago.

    """)



    # 2. Add a subheader to make the list look more official.

    st.subheader("Dashboard Features")

    st.markdown("""

    - **Data Overview:** See high-level charts and trends in the data.

    - **Crime Heatmap:** Visualize the geographical hotspots of crime.

    - **Crime Type Prediction:** Predict the type of crime based on location and time.

    - **Crime Count Forecast:** Forecast the total number of crimes for the next month.

    - **Video Object Detection:** Analyze a video to detect objects like people and vehicles.

    """)



# --- PAGE 2: DATA OVERVIEW ---

elif page == "Data Overview":

    st.title("Data Overview and Trends")

    st.write("This page shows high-level statistics and charts about the crime data.")



    st.divider()



    # Create two columns for charts

    col1, col2 = st.columns(2)



    with col1:

        # 1. Top 10 Crime Types

        st.subheader("Top 10 Crime Types")

        top_crimes = df['Primary Type'].value_counts().head(10)

        st.bar_chart(top_crimes)



        st.divider()



        # 2. Crimes by Day of Week

        st.subheader("Crimes by Day of Week")

        crimes_by_day = df['DayOfWeek'].value_counts().reindex([

            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'

        ])

        st.bar_chart(crimes_by_day)



    with col2:

        # 3. Crimes by Hour of Day

        st.subheader("Crimes by Hour of Day")

        crimes_by_hour = df['Hour'].value_counts().sort_index()

        st.area_chart(crimes_by_hour)



    st.divider()



    # 4. Historical Monthly Chart

    st.subheader("Historical Monthly Crime Counts")

    df_lstm = df.set_index('Date').resample('ME').size().to_frame('CrimeCount').iloc[:-1]

    st.line_chart(df_lstm)





# --- PAGE 3: CRIME HEATMAP ---

elif page == "Crime Heatmap":

    st.title("Crime Hotspot Heatmap")

    st.write("This map shows the concentration of reported crimes across Chicago. Red areas indicate higher crime density.")

    

    # Fixed 'utf-8' encoding

    with open('./outputs/crime_heatmap.html', 'r', encoding='utf-8') as f:

        heatmap_html = f.read()

    

    # Fixed `st.components.v1.html`

    st.components.v1.html(heatmap_html, height=600)





# --- PAGE 4: CRIME TYPE PREDICTION (FFN) ---

elif page == "Crime Type Prediction":

    st.title("Predict the Type of Crime")

    st.write("Enter the details below to predict the most likely crime type.")



    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            latitude = st.number_input("Latitude", value=41.8781, format="%.4f")

            longitude = st.number_input("Longitude", value=-87.6298, format="%.4f")

        with col2:

            hour = st.slider("Hour of Day (0-23)", 0, 23, 12)

            month = st.slider("Month (1-12)", 1, 12, 6)

        with col3:

            day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)



        submit_button = st.form_submit_button(label="Predict Crime Type")



    st.divider()



    if submit_button:

        with st.spinner("Analyzing..."):

            input_data = np.array([[latitude, longitude, hour, month, day_of_week]])

            scaled_data = ffn_scaler.transform(input_data)

            prediction = ffn_model.predict(scaled_data)

            predicted_class_index = np.argmax(prediction)

            predicted_crime = ffn_encoder.inverse_transform([predicted_class_index])[0]

            

            st.success(f"Predicted Crime Type: **{predicted_crime}**")





# --- PAGE 5: CRIME COUNT FORECAST (LSTM) ---

elif page == "Crime Count Forecast":

    st.title("Monthly Crime Count Forecast")

    st.write("This section forecasts the total number of crimes for the next month.")

    

    if st.button("Forecast Next Month's Crime Count"):

        with st.spinner("Forecasting..."):

            df_lstm = df.set_index('Date').resample('ME').size().to_frame('CrimeCount').iloc[:-1]

            last_12_months = df_lstm['CrimeCount'].values[-12:].reshape(-1, 1)

            scaled_last_12 = lstm_scaler.transform(last_12_months)

            input_data_lstm = scaled_last_12.reshape((1, 12, 1))

            

            prediction_scaled = lstm_model.predict(input_data_lstm)

            prediction = lstm_scaler.inverse_transform(prediction_scaled)

            

            st.success(f"Forecasted Crime Count for Next Month: **{int(prediction[0][0])}**")





# --- PAGE 6: VIDEO OBJECT DETECTION (YOLO) ---

elif page == "Video Object Detection":

    st.title("Real-time Object Detection in Video")

    st.write("Upload a video file to detect objects like people, cars, and backpacks.")



    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    

    st.divider()



    if uploaded_file is not None:

        st.video(uploaded_file)

        

        if st.button("Process Video"):

            temp_file_path = os.path.join("temp_video.mp4")

            with open(temp_file_path, "wb") as f:

                f.write(uploaded_file.getbuffer())



            cap = cv2.VideoCapture(temp_file_path)

            

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fps = int(cap.get(cv2.CAP_PROP_FPS))

            

            # Fixed video codec 'avc1' for .mp4

            fourcc = cv2.VideoWriter_fourcc(*'avc1')

            out_path = os.path.join("processed_video.mp4")

            out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))



            with st.spinner("Processing video... This may take a while."):

                while cap.isOpened():

                    success, frame = cap.read()

                    if not success:

                        break

                    results = yolo_model.track(frame, persist=True)

                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)



            cap.release()

            out.release()

            st.success("Video processing complete!")



            st.video(out_path)

            with open(out_path, "rb") as file:

                st.download_button(

                    label="Download Processed Video",

                    data=file,

                    file_name="yolo_output.mp4",

                    mime="video/mp4"

                )