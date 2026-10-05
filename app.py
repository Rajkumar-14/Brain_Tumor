import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect tumor presence and type.")

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

@st.cache_resource
def load_tumor_model():
    path = "models/model.h5" if os.path.exists("models/model.h5") else "model.h5"
    return tf.keras.models.load_model(path)

with st.spinner("Loading model..."):
    model = load_tumor_model()

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing scan..."):
            img = image.resize((128, 128))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_index = int(np.argmax(prediction, axis=1)[0])
            confidence = float(np.max(prediction, axis=1)[0]) * 100
            label = class_labels[predicted_index]

            st.divider()
            if label == 'notumor':
                st.success("No Tumor Detected")
            else:
                st.error(f"Tumor Detected: {label.capitalize()}")
            st.info(f"Confidence: {confidence:.2f}%")
