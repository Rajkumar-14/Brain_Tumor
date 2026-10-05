import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect if there is a tumor and its type.")

# Class labels matching training order
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Cache model loading so it doesn't reload on every interaction
@st.cache_resource
def load_prediction_model():
    model_path = "models/model.h5" if os.path.exists("models/model.h5") else "model.h5"
    return tf.keras.models.load_model(model_path)

model = load_prediction_model()

uploaded_file = st.file_uploader(
    "Select MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Upload and Detect", type="primary"):
        with st.spinner("Analyzing image..."):
            # Target size matching main.py (128, 128)
            img = image.resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_index = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions, axis=1)[0]) * 100

            predicted_label = class_labels[predicted_index]

            st.divider()
            if predicted_label == "notumor":
                st.success(f"**Result:** No Tumor detected")
            else:
                st.error(f"**Result:** Tumor: {predicted_label.capitalize()}")

            st.info(f"**Confidence:** {confidence:.2f}%")
