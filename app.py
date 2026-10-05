import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🧠 Brain Tumor Detection")

model = tf.keras.models.load_model("model.h5")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI")

    if st.button("Predict"):

        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        st.write("Prediction:", prediction)