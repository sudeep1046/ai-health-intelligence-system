# AI-Powered Multimodal Personal Health Intelligence System 🧠🍎

This project is an end-to-end, offline AI-powered health intelligence system designed to analyze food images and estimate calorie intake using deep learning techniques. The system is built without relying on external APIs and focuses on privacy-preserving, on-device intelligence.

## 🚀 Features
- Food calorie estimation using a CNN-based image classification model
- Real-time inference through a FastAPI backend
- Interactive Streamlit frontend for image upload and prediction
- Confidence scoring for predictions
- Modular and scalable architecture
- Offline execution with no external API dependency

## 🛠️ Tech Stack
- Python 3.11
- TensorFlow / Keras
- Scikit-learn
- OpenCV
- FastAPI
- Uvicorn
- Streamlit

## 📂 Project Overview
The system allows users to upload food images through a web interface. These images are processed and sent to a backend service where a trained convolutional neural network predicts the food category. Based on the prediction and user-provided portion size, the system estimates calorie intake and displays the results in real time.

## ▶️ How to Run
1. Create a virtual environment  
2. Install dependencies from `requirements.txt`  
3. Start the FastAPI backend  
4. Launch the Streamlit frontend  
5. Upload a food image and view predictions  

## 👤 Author
**Sudeep**  
AI & Machine Learning Enthusiast
