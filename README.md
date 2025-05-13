# 🔍 Anomaly Inspector

![Anomaly Inspector](https://img.shields.io/badge/Anomaly-Inspector-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

An AI-powered quality control application that uses computer vision to detect anomalies in products. Built with TensorFlow and Streamlit to provide an intuitive, user-friendly interface for real-time anomaly detection.

## ✨ Features

- **Dual Input Methods**: Upload images or use live webcam feed for real-time detection
- **Real-time Analysis**: Instant feedback with confidence scores for detected anomalies
- **User-friendly Interface**: Clean, intuitive design with clear visual feedback
- **High Accuracy**: Powered by a custom TensorFlow model trained on product images

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection feature)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/nishil61/Anomaly-Inspector.git
   cd Anomaly-Inspector
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the model**
   
   The model file is too large for GitHub. Download it from [this Google Drive link](https://drive.google.com/your-model-link) and place it in the `model/` directory with the filename `teachable_machine_model.h5`.

## 🏃‍♂️ Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`.

## 📊 Creating Your Own Model with Teachable Machine

You can create your own custom anomaly detection model using Google's Teachable Machine:

1. **Visit Teachable Machine**
   - Go to [teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/)
   - Click on "Get Started" and select "Image Project" → "Standard Image Model"

2. **Prepare Your Dataset**
   - Create two classes: "Good" and "Anomaly"
   - For "Good" class: Upload 30+ images of products without defects
   - For "Anomaly" class: Upload 30+ images of products with defects/anomalies
   - Ensure images have varied lighting, angles, and positions for better generalization

3. **Train Your Model**
   - Click the "Train Model" button
   - Wait for training to complete (usually takes 1-3 minutes)

4. **Export Your Model**
   - Click "Export Model"
   - Select "Tensorflow" → "Keras"
   - Download the model file (.h5 format)
   - Place the downloaded model in the `model/` directory with the name `teachable_machine_model.h5`

5. **Test Your Custom Model**
   - Restart the Anomaly Inspector application
   - Your custom model will be automatically loaded

## 🖼️ Application Structure

- `app.py`: Main application file with Streamlit UI and logic
- `model/`: Directory containing the trained neural network model
- `docs/`: Documentation and images
- `requirements.txt`: Required Python packages

## 💻 Usage Guide

### Upload Image Mode
1. Select "Upload Image" from the radio buttons
2. Click "Browse files" to upload an image
3. After successful upload, click "Submit for Anomaly Detection"
4. View the analysis results with confidence scores

### Live Feed Mode
1. Select "Live Feed" from the radio buttons
2. Click "Start Live Feed" to activate your webcam
3. Position the object in the camera view
4. View real-time anomaly detection results
5. Click "Stop Live Feed" when finished

## 🔧 Tech Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: Pillow, NumPy

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/) for the machine learning framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Google Teachable Machine](https://teachablemachine.withgoogle.com/) for model training platform
- [OpenCV](https://opencv.org/) for computer vision functionality

## 📧 Contact

For questions or feedback, please reach out to [nishilvalia61@gmail.com](mailto:pathaknishil3642@gmail.com)
