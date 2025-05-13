import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image

# Page configuration - similar to InspectorsAlly
st.set_page_config(page_title="Anomaly Inspector", page_icon="🔍", layout="wide")

# Apply custom CSS for better UI
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0 !important;
    }
    
    /* Caption styling */
    .caption-text {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
    
    /* Radio button styling */
    .stRadio > div {
        padding: 10px;
        margin-bottom: 20px;
    }
    
    /* Progress bars styling */
    .stProgress > div > div {
        height: 20px;
        border-radius: 10px;
    }
    .good-progress > div > div {
        background-color: #4CAF50 !important;
    }
    .anomaly-progress > div > div {
        background-color: #F44336 !important;
    }
    
    /* Info containers */
    .info-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Success message */
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Warning message */
    .warning-msg {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Custom DepthwiseConv2D to ignore unsupported arguments
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove unsupported 'groups' argument
        super().__init__(*args, **kwargs)

# Load the model with the custom layer
@st.cache_resource
def load_model_once():
    model_path = "model/teachable_machine_model.h5"
    model = load_model(model_path, compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print("Model loaded successfully!")
    return model

# Load model
model = load_model_once()

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    return image

# Function to detect anomaly
def detect_anomaly(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)[0]
    confidence_good = predictions[0] * 100
    confidence_anomaly = predictions[1] * 100
    predicted_class = "Good" if confidence_good > confidence_anomaly else "Anomaly"
    return confidence_good, confidence_anomaly, predicted_class

# Initialize session state for persistent tab selection
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # Default to first tab

# Main header - styled like InspectorsAlly
st.markdown("<h1 class='main-header'>Anomaly Inspector</h1>", unsafe_allow_html=True)
st.markdown("<p class='caption-text'>Boost Your Quality Control with Anomaly Inspector - The Ultimate AI-Powered Inspection App</p>", unsafe_allow_html=True)

# Description like InspectorsAlly
st.write("Try showing leather image in Live Feed or uploading it and watch how an AI Model will classify it between Good / Anomaly.")

# Create sidebar with info - like InspectorsAlly
with st.sidebar:
    # Add image at the top (assuming the file exists)
    try:
        img = Image.open("./docs/overview_dataset.jpg")
        st.image(img, use_container_width=True)
    except FileNotFoundError:
        st.error("Image not found: ./docs/overview_dataset.jpg")
        st.info("Please add the overview image to the docs folder.")
    
    # Add a green background container for sidebar text
    st.markdown("""
    <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h3 style="color: #155724;">About Anomaly Inspector</h3>
        <p style="color: #155724;">
            Anomaly Inspector is a powerful AI-powered application designed to help businesses streamline their quality control inspections. 
            With Anomaly Inspector, companies can ensure that their products meet the highest standards of quality, 
            while reducing inspection time and increasing efficiency.
        </p>
        <p style="color: #155724;">
            This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models 
            to perform visual quality control inspections with unparalleled accuracy and speed. 
            Anomaly Inspector is capable of identifying even the slightest defects, such as scratches, dents, 
            discolorations, and more on the product images.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Selection method UI - like InspectorsAlly
st.subheader("Select Image Input Method")
current_tab = st.radio(
    "options", ["Upload Image", "Live Feed"], 
    index=st.session_state.active_tab,
    label_visibility="collapsed"
)

# Update the active tab in session state
for i, title in enumerate(["Upload Image", "Live Feed"]):
    if current_tab == title:
        st.session_state.active_tab = i

# Upload Image Tab (File Uploader) - like InspectorsAlly
if current_tab == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"], key="image_uploader"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        st.markdown("<div class='success-msg'>Image uploaded successfully!</div>", unsafe_allow_html=True)
        
        # Add a submit button like InspectorsAlly (with green color)
        submit = st.button(
            label="Submit for Anomaly Detection", 
            type="primary",  # This will make it green
            use_container_width=True
        )
        
        if submit:
            # Convert to OpenCV format
            image = np.array(image)
            
            # Detect anomaly
            with st.spinner(text="This may take a moment..."):
                confidence_good, confidence_anomaly, predicted_class = detect_anomaly(image)
            
            # Display results in a way similar to InspectorsAlly
            st.subheader("Output")
            
            # Craft prediction sentence like InspectorsAlly
            if predicted_class == "Good":
                prediction_sentence = "Congratulations! Your product has been classified as a 'Good' item with no anomalies detected in the inspection images."
                st.success(prediction_sentence)
            else:
                prediction_sentence = "We're sorry to inform you that our AI-based visual inspection system has detected an anomaly in your product."
                st.error(prediction_sentence)
            
            # Show confidence scores
            st.subheader("Confidence Scores:")
            st.markdown("<div class='good-progress'>", unsafe_allow_html=True)
            st.progress(float(confidence_good)/100, text=f"Good: {confidence_good:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='anomaly-progress'>", unsafe_allow_html=True)
            st.progress(float(confidence_anomaly)/100, text=f"Anomaly: {confidence_anomaly:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-msg'>Please upload an image file.</div>", unsafe_allow_html=True)

# Live Feed Tab (Camera Input) - like InspectorsAlly
elif current_tab == "Live Feed":
    st.markdown("<div class='warning-msg'>Please allow access to your camera.</div>", unsafe_allow_html=True)
    
    # Create columns for better button placement
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    
    # Session state to track if the camera is running
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    # Add stylish button in the middle column
    with button_col2:
        if not st.session_state.camera_running:
            button_start = st.button("▶️ Start Live Feed", 
                          use_container_width=True, 
                          type="primary",
                          help="Start your webcam for real-time anomaly detection")
            if button_start:
                st.session_state.camera_running = True
                st.rerun()
        else:
            button_stop = st.button("⏹️ Stop Live Feed", 
                         use_container_width=True,
                         type="secondary", 
                         help="Stop the webcam")
            if button_stop:
                st.session_state.camera_running = False
                st.rerun()
    
    # Create placeholders for video feed and results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
    
    with col2:
        # Create a container for results that can be cleared
        results_placeholder = st.container()
        with results_placeholder:
            results_title = st.empty()
            prediction_text = st.empty()
            good_container = st.empty()
            anomaly_container = st.empty()
            
        # Create a separate container for info messages
        info_container = st.container()
    
    # Use session state to control camera
    if st.session_state.camera_running:
        # Use OpenCV to capture webcam feed
        cap = cv2.VideoCapture(0)
        
        # Display the title once
        results_title.subheader("Real-time Analysis")
        
        # Clear the info container and add the new message
        info_container.empty()
        info_message = info_container.info("Click 'Stop Live Feed' button to end real-time anomaly detection")
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                st.session_state.camera_running = False
                break
                
            # Process frame for anomaly detection
            confidence_good, confidence_anomaly, predicted_class = detect_anomaly(frame)
            
            # Add prediction text to frame with color-coding
            color = (0, 255, 0) if predicted_class == "Good" else (0, 0, 255)  # Green for Good, Red for Anomaly
            cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Show the frame
            video_placeholder.image(frame, channels="BGR", caption="Live Feed", use_container_width=True)
            
            # Update the results in real-time
            if predicted_class == "Good":
                prediction_text.success("Congratulations! Your product has been classified as a 'Good' item.")
            else:
                prediction_text.error("We're sorry, our system has detected an anomaly.")
            
            # Display progress bars
            good_container.progress(float(confidence_good)/100, text=f"Good: {confidence_good:.2f}%")
            anomaly_container.progress(float(confidence_anomaly)/100, text=f"Anomaly: {confidence_anomaly:.2f}%")
        
        # Release webcam when stopped
        cap.release()
    else:
        # Display a camera-off message
        with video_placeholder:
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; height: 300px; 
                        background-color: #f8f9fa; border-radius: 5px; margin-top: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 40px; margin-bottom: 10px;">📷</div>
                    <p style="font-size: 18px; color: #6c757d;">Camera is currently off</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Clear any previous message in the info container
        info_container.empty()
        
        # Display instruction when camera is off
        with info_container:
            st.info("Click 'Start Live Feed' button to begin real-time anomaly detection")

# Footer
st.markdown("---")
st.markdown("Anomaly Inspector v1.0 | Made with Streamlit and TensorFlow by Nishil ❤️")