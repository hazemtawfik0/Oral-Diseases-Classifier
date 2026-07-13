# 🦷 Oral Diseases Image Classifier

This project is a deep learning-based web application designed to classify various oral diseases using a pre-trained **ResNet50** model. It provides an automated, efficient way to assist in the early detection of oral health issues by analyzing images.

## 🚀 Key Features
- **Deep Learning Model:** Built using **Transfer Learning (ResNet50)** for high accuracy.
- **Data Handling:** Trained on an imbalanced dataset using **Class Weights** to ensure fair classification across all disease types.
- **Web Interface:** A user-friendly web app powered by **Flask**, allowing users to upload images and get real-time predictions.
- **Scalable:** Easy to integrate into larger healthcare diagnostic systems.

## 🛠️ Tech Stack
- **Languages:** Python
- **Frameworks:** Flask (Backend), TensorFlow/Keras (DL Model)
- **Data Processing:** NumPy, Pillow, Scikit-learn
- **Deployment:** Local Server / Git

## 📂 Project Structure
```text
/app
├── app.py                  # Main Flask application
├── oral_disease_resnet50.rar # Trained model (compressed)
├── requirements.txt        # Project dependencies
└── static/                 # CSS and Images
└── templates/              # HTML files