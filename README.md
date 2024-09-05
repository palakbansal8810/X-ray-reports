# X-ray-reports


# Medical Imaging Report Generator

## Overview

This project provides a comprehensive tool for analyzing medical images using pre-trained models. It generates diagnostic reports based on chest X-rays, bone fracture images, and brain tumor images. The tool integrates Grad-CAM visualization for chest X-rays to highlight regions of interest and uses AI for generating textual reports.

## Features

- **Predict Bone Fracture**: Classify bone X-ray images to determine if a fracture is present.
- **Predict Brain Tumor**: Determine if a brain tumor is present in an image.
- **Analyze Chest X-rays**: Classify chest X-rays into various categories and visualize the results using Grad-CAM.
- **Generate Reports**: Create detailed reports based on predictions using generative AI.

## Requirements

- Python 3.6 or higher
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Pandas
- NumPy
- Google Generative AI API

## Installation

1. **Clone the repository**:
   ```bash
   git clone 'https://github.com/palakbansal8810/X-ray-reports.git'
   ```

2. **Install the required packages**:
   ```bash
   pip install tensorflow keras opencv-python matplotlib pandas numpy google-generativeai
   ```

## Usage

1. **Prepare Models**:
   - Ensure the pre-trained models (`fracture.keras`, `brain_tumor_dataset.keras`, `chestXray_model.keras`) are available in the `models` directory.

2. **Run the Script**:
   Execute the script by running:
   ```bash
   python main.py
   ```

3. **Provide Inputs**:
   - Enter the patient's ID.
   - Provide the file paths for the bone X-ray, chest X-ray, and brain tumor images when prompted.

4. **Review Outputs**:
   - The script will output a bar graph of chest X-ray predictions and a Grad-CAM visualization of the chest X-ray.
   - A diagnostic report will be generated and printed based on the provided predictions.

## Code Description

- **`get_weighted_loss`**: Function to create a custom loss function with positive and negative weights.
- **`load_image_normalize`**: Function to load and normalize an image.
- **`grad_cam`**: Function to generate a Grad-CAM heatmap for visualizing model decisions.
- **`compute_gradcam`**: Function to compute and save Grad-CAM visualizations for selected labels.
- **`predict_fractured`**: Function to predict if a bone is fractured.
- **`predict_tumor`**: Function to predict if a brain tumor is present.
- **`get_gemini_report`**: Function to generate a report using Google Generative AI.

X-ray-reports/ ├── models/ │ ├── brain_tumor_dataset.keras │ ├── chestXray_model.keras │ ├── fracture.keras │ └── densenet.hdf5 ├── outputs/ │ ├── bargraph.png │ └── gradcam.png ├── examples/ │ └── example ├── main.py ├── ChestXRay_Medical_Diagnosis_Deep_Learning (1).ipynb ├── brain-tumor-detection-binary-classification.ipynb ├── bone-x-ray-imaging-classification-and-analysis.ipynb ├── sample_labels.csv ├── dataset.csv └── README.md

## Example

To generate a report:

1. **Enter Inputs**:
   ```
   Enter patient's id
   12345
   Enter bone x-ray path:
   'examples\bone x-ray\bone3.png'
   Enter chest x-ray image path:
   'examples\chest-xray\00000473_000.png'
   Enter a brain x-ray image path:
   'examples\brain_tumor\Y250.jpg'
   ```

2. **Review Results**:
   - Check the `outputs` directory for the bar graph and Grad-CAM visualizations.
   - Read the generated report from the console output.

