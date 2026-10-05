import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect if there is a tumor and its type.")

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

MODEL_FILE = "model.h5"
# Direct download link from your GitHub Release
MODEL_URL = "https://github.com/rajkumar-14/brain_tumor/releases/download/v1.0/model.h5"

@st.cache_resource
def load_tumor_model():
    # Download the model if it's not present locally
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading trained model weights (first load only)..."):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_FILE, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"Failed to download model from GitHub release (HTTP {response.status_code}).")
                st.stop()

    return tf.keras.models.load_model(MODEL_FILE)

model = load_tumor_model()

uploaded_file = st.file_uploader(
    "Select MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Upload and Detect", type="primary"):
        with st.spinner("Analyzing image..."):
            img = image.resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_idx = int(np.argmax(predictions, axis=1)[0])
            confidence = float(np.max(predictions, axis=1)[0]) * 100

            label = class_labels[predicted_idx]

            st.divider()
            if label == "notumor":
                st.success("Result: No Tumor detected")
            else:
                st.error(f"Result: Tumor detected ({label.capitalize()})")

            st.info(f"Confidence: {confidence:.2f}%")
