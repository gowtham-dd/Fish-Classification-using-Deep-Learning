import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Title and UI ---
st.set_page_config(page_title="Fish Classifier", layout="centered")
st.title("🐟 Fish Classifier App")
st.markdown("Upload a **fish image** and select a **model** to predict its class along with confidence scores.")

# --- Model files ---
model_files = {
    "VGG16": "D:/Data Science/Fish_Classification/VGG16.h5",
    "ResNet50 v3": "D:/Data Science/Fish_Classification/ResNet50_v3.h5",
    "MobileNetV2": "D:/Data Science/Fish_Classification/MobileNetV2.h5",
    "InceptionV3": "D:/Data Science/Fish_Classification/InceptionV3.h5",
}

# --- Class labels ---
label_map = {
    0: "Black Sea Sprat",
    1: "Gilt Head Bream",
    2: "Horse Mackerel",
    3: "Red Mullet",
    4: "Red Sea Bream",
    5: "Sea Bass",
    6: "Shrimp",
    7: "Striped Red Mullet",
    8: "Trout"
}

# --- Dropdown to choose model ---
model_choice = st.selectbox("Choose a model", list(model_files.keys()))

# --- File uploader ---
uploaded_image = st.file_uploader("Upload a fish image", type=['jpg', 'jpeg', 'png'])

# --- Load model with caching ---
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# --- If image is uploaded ---
if uploaded_image is not None and model_choice:
    model = load_model(model_files[model_choice])

    # Preprocess image
    image = Image.open(uploaded_image).convert('RGB')
    image_resized = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)[0]
    predicted_class_idx = np.argmax(predictions)
    predicted_label = label_map.get(predicted_class_idx, "Unknown")
    confidence = predictions[predicted_class_idx]

    # Display results
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    st.success(f"🎯 **Predicted Class:** {predicted_label}")
    st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")

    # Plot beautiful horizontal bar chart
    st.subheader("🔍 Confidence Scores")
    fig, ax = plt.subplots(figsize=(8, 5))
    class_names = [label_map[i] for i in range(len(predictions))]
    bars = ax.barh(class_names, predictions, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Model Confidence for Each Class")
    ax.invert_yaxis()

    # Add confidence score as text
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f"{width*100:.1f}%", va='center')

    st.pyplot(fig)
