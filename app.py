import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

import os

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "dr_model_final.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

st.title("Diabetic Retinopathy Detection System")
st.write("Upload a retinal image to predict the DR stage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0])
    plt.xticks(rotation=45)
    st.pyplot(fig)
