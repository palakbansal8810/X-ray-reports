import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2 
import math
import os
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import img_to_array
import keras
import google.generativeai as genai

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += - pos_weights[i] * K.mean(y_true[:, i] * K.log(y_pred[:, i] + epsilon)) \
                   - neg_weights[i] * K.mean((1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
        return loss
    return weighted_loss

def load_image_normalize(path, mean, std, H=320, W=320):
    x = image.load_img(path, target_size=(H, W))
    x = image.img_to_array(x)  # Convert image to array
    x -= mean
    x /= std
    x = np.expand_dims(x, axis=0)
    return x

def load_image(path, preprocess=True, mean=None, std=None, H=320, W=320):
    x = image.load_img(path, target_size=(H, W))
    x = image.img_to_array(x)
    if preprocess and mean is not None and std is not None:
        x -= mean
        x /= std
    x = np.expand_dims(x, axis=0)
    return x

# GradCAM function
def grad_cam(input_model, image, category_index, layer_name):

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    grad_model = tf.keras.models.Model(
        inputs=[input_model.inputs],
        outputs=[input_model.get_layer(layer_name).output, input_model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        predictions = tf.squeeze(predictions)
        
        if category_index >= predictions.shape[0]:
            raise ValueError(f"category_index {category_index} is out of bounds for predictions shape {predictions.shape}")
        
        loss = predictions[category_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)
    
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    
    return heatmap

def compute_gradcam(model, img, mean, std, labels, selected_labels, layer_name='conv5_block16_concat', gradcam_save_path='outputs'):
   
    img_path =  img
    preprocessed_input = load_image_normalize(img_path, mean, std)
    predictions = model.predict(preprocessed_input)
    predictions = np.squeeze(predictions)
    
    num_labels = len(selected_labels)
    grid_size = int(math.ceil(math.sqrt(num_labels)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    original_image = load_image(img_path, preprocess=False)
    axes[0, 0].imshow(original_image[0] / 255.0, cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')
    
    index = 1
    
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            
            print(f"Generating GradCAM for class {labels[i]} (p={predictions[i]:.2f})")
            
            row = index // grid_size
            col = index % grid_size
            
            axes[row, col].imshow(original_image[0] / 255.0, cmap='gray')
            axes[row, col].imshow(gradcam, cmap='magma', alpha=min(0.56, predictions[i]))
            
            axes[row, col].set_title(f"{labels[i]}: {predictions[i]:.3f}")
            axes[row, col].axis('off')
            
            index += 1
    
    for i in range(index, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if not os.path.exists(gradcam_save_path):
        os.makedirs(gradcam_save_path)
    
    plt.savefig(f'{gradcam_save_path}/gradcam.png')

model_fracture = keras.models.load_model('models/fracture.keras')

def predict_fractured(model, image_path):
   
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    prediction = model.predict(img_array)
    print(f'Raw prediction output of fracture: {prediction}')  # Debugging line

    probability = prediction[0][0]
    class_label = 'The bone is fractured' if probability >= 0.2 else 'The bone is not fractured'
    return class_label, probability

model_brain=keras.models.load_model('models/brain_tumor_dataset.keras')

def predict_tumor(img_path,model):
 
    img = cv2.imread(img_path)
    if img is None:
      print(f"Error: Could not load image at '{img_path}'")
      return None

    resize = tf.image.resize(img, (256, 256))
    predict = model.predict(np.expand_dims(resize/255, 0))
    if predict > 0.5:
      result="The person has Brain Tumor"
      print(result)
      return result,predict
    else:
      result="The Person Does not have Brain Tumor"
      print(result)
      return result,predict

def get_gemini_report(predictions, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        response = model.generate_content([predictions, prompt])
        return response.text
    except google.api_core.exceptions.GoogleAPIError as e:
        print(f"An error occurred: {e}")
        return "Error generating report. Please try again later."

if __name__ == "__main__":
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
          'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    mean, std = 116.907412109375, 59.33966556953607
    
    neg_weights = np.array([0.05431548, 0.0483631, 0.2485119, 0.00297619, 0.38541667,
                            0.11309524, 0.12425595, 0.18973214, 0.10714286, 0.06845238,
                            0.02752976, 0.03497024, 0.0453869, 0.08630952])

    pos_weights = np.array([0.94568452, 0.9516369, 0.7514881, 0.99702381, 0.61458333,
                            0.88690476, 0.87574405, 0.81026786, 0.89285714, 0.93154762,
                            0.97247024, 0.96502976, 0.9546131, 0.91369048])

    weighted_loss = get_weighted_loss(pos_weights, neg_weights)

    model_chest = load_model('models/chestXray_model.keras', custom_objects={'weighted_loss': weighted_loss})

    genai.configure(api_key='AIzaSyCRBedlGebhqf-BFqR3VAZwASNlz6afaqY')
    patient_id=input("Enter patient's id\n")
    image_bone_xray = input('Enter bone x-ray path:\n')
    image_path_for_chest_xray = input('Enter chest x-ray image path:\n')
    image_path_for_tumour=input('Enter a brain x-ray image path:\n')
    
    label_fr, probability_fr = predict_fractured(model_fracture, image_bone_xray)

    processed_image = load_image_normalize(image_path_for_chest_xray, mean, std)
    preds = model_chest.predict(processed_image)
    print(f'chest: {preds}')
    result_chest=[]
    for i in range(len(labels)):
        ans=f'{labels[i]}: {preds[0][i]}'
        result_chest.append(ans)
    pred_df = pd.DataFrame(preds, columns=labels)
    pred_df.loc[0, :].plot.bar()

    plt.title("Predictions")
    plt.tight_layout()
    plt.savefig('outputs/bargraph.png')
    
    compute_gradcam(model_chest, image_path_for_chest_xray, mean, std, labels, labels)
    result_tu,prediction_tu=predict_tumor(image_path_for_tumour,model_brain)
    input_prompt = f"""
    As an experienced physician specializing in radiology and diagnostic imaging, your task is to generate a comprehensive report based on the following diagnostic results:
    Referring Physician: Da Vinci's doctor
    1. **Chest X-ray Results**: {result_chest}
    2. **Brain Tumor Diagnosis**: {result_tu} (Prediction: {prediction_tu})
    3. **Bone Fracture Evaluation**: {label_fr} (Probability: {probability_fr})

    Please provide a detailed analysis of these findings, integrating them into a cohesive report. Ensure to address any potential correlations between the conditions observed and offer professional insights or recommendations for further action. 

    Include the patient's ID: {patient_id} in the report.

    Your expertise and thorough evaluation are crucial in assisting with the accurate diagnosis and treatment plan.
    """
    predictions=''
    response = get_gemini_report(predictions, input_prompt)
    print(response)
   

