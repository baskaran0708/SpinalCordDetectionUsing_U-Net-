# SpineAI: Automated Spinal Analysis System

![SpineAI Banner](https://img.shields.io/badge/SpineAI-Deep_Learning-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 🏥 Project Overview
SpineAI is a deep learning-based system designed to automate the analysis of spinal X-rays. It assists radiologists and orthopedic specialists by automatically detecting the view type, segmenting vertebral regions, and analyzing spinal curvature for potential abnormalities like scoliosis.

## 🎯 Purpose & Impact
The primary goal of this project is to enhance the efficiency and accuracy of spinal diagnostics.
- **Automated Triage**: Quickly identifies key spinal regions (Cervical, Thoracic, Lumbar).
- **Scoliosis Detection**: Mathematically analyzes the spine's shape to detect S-curves or C-curves.
- **Support Tool**: Acts as a second pair of eyes for medical professionals, reducing fatigue and oversight.

## 🧠 Technical Architecture
The system employs state-of-the-art computer vision models:
1.  **View Classifier**: Uses **EfficientNet-B0** to distinguish between AP (Anterior-Posterior) and Lateral views.
2.  **Region Segmentor**: Uses **U-Net++** with an **EfficientNet-B3** backbone to precisely segment spinal regions.
3.  **Shape Analyzer**: A post-processing algorithm that fits polynomial curves to the segmented spine to detect curvature anomalies.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Installation
1.  **Clone the repository**
    ```bash
    git clone https://github.com/baskaran0708/SpinalCordDetectionUsing_U-Net-.git
    cd SpinalCordDetectionUsing_U-Net-
    ```

2.  **Set up Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1.  **Training**
    - To train the classifier: `python train_classifier.py`
    - To train the segmentor: `python train_segmentor.py`

2.  **Inference**
    - Run the main pipeline on an image:
    ```python
    from main_pipeline import analyze_patient
    analyze_patient("path/to/xray.png")
    ```

## 🤝 How to Contribute
We welcome contributions from the open-source community! Here's how you can help:

1.  **Fork the Project**: Create your own copy of the repository.
2.  **Create a Branch**: `git checkout -b feature/AmazingFeature`
3.  **Commit Changes**: `git commit -m 'Add some AmazingFeature'`
4.  **Push to Branch**: `git push origin feature/AmazingFeature`
5.  **Open a Pull Request**: Submit your changes for review.

### Areas for Contribution
- **Data Collection**: We need more diverse X-ray datasets (especially lateral views).
- **Model Optimization**: Experiment with lighter backbones for mobile deployment.
- **UI Development**: Build a web-based dashboard for doctors.

## 💖 Support the Project
If you find this project useful, please consider:
- **Starring** the repository on GitHub.
- **Sharing** it with researchers and medical professionals.
- **Reporting Issues** to help us improve.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Disclaimer: This tool is for research purposes only and should not be used as the sole basis for medical diagnosis.*
