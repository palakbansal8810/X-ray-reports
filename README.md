# X-ray-reports

Sure! Here’s a README file for your project that explains its functionality, setup, and usage:

---

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
   git clone <repository-url>
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
   python <script-name>.py
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

## Example

To generate a report:

1. **Enter Inputs**:
   ```
   Enter patient's id
   12345
   Enter bone x-ray path:
   /path/to/bone_xray.jpg
   Enter chest x-ray image path:
   /path/to/chest_xray.jpg
   Enter a brain x-ray image path:
   /path/to/brain_xray.jpg
   ```

2. **Review Results**:
   - Check the `outputs` directory for the bar graph and Grad-CAM visualizations.
   - Read the generated report from the console output.

## Troubleshooting

- Ensure all file paths are correct and files are accessible.
- Check API keys and model paths if errors occur during report generation.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to adjust or add any information based on your specific requirements and details!
