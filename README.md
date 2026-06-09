#  NeuroVision AI

Brain Tumor Segmentation using **ResUNet** with **Grad-CAM Explainability** and **AI-generated PDF Medical Reports**.

---

##  Overview

NeuroVision AI is a deep learning-based application that performs brain tumor segmentation from MRI images. The system uses a ResUNet architecture to localize tumor regions and provides visual explanations using Grad-CAM. A Gradio-based interface allows users to upload MRI images and receive analysis results along with downloadable reports.

---

## Features

* Brain Tumor Segmentation
* Overlay Visualization
* Grad-CAM Explainability
* Confidence Estimation
* Severity Level Analysis
* AI-generated PDF Medical Reports
* Download Predicted Mask
* Download Overlay Image
* Interactive Gradio Interface

---

##  Model

* Architecture: **ResUNet**
* Framework: **TensorFlow / Keras**
* Dataset: **LGG MRI Segmentation Dataset**

---

##  Performance

| Metric           | Score  |
| ---------------- | ------ |
| Dice Coefficient | 0.8587 |
| IoU Score        | 0.7563 |

---

## Application Output

The application provides:

* Original MRI Image
* Predicted Tumor Mask
* Overlay Visualization
* Grad-CAM Heatmap
* Tumor Information
* Confidence Score
* Severity Level
* Downloadable PDF Report

---

##  Repository Structure

```text
Brain-Tumor-Segmentation-ResUNet
│
├── app.py
├── requirements.txt
├── README.md
├── notebooks
      ├── Brain_tumor_segmentation.ipynb
      └── Brain_tumor_gradio.ipynb
```

---

##  Model Download

The trained ResUNet model (~380 MB) exceeds GitHub's upload limit.

Download the model from:

**https://drive.google.com/file/d/1MhwoC6SNDFF9vqNkPddr5_zk0J4uq1jX/view?usp=drive_link**

After downloading, place the model inside:

```text
model/resunet_brain_tumor.keras
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Brain-Tumor-Segmentation-ResUNet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* Gradio
* Matplotlib
* ReportLab



## ⚠ Disclaimer

This application is intended for educational and research purposes only and should not be considered a substitute for professional medical diagnosis or treatment.
