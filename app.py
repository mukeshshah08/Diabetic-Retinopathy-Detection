import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Rebuild Model Architecture
# -----------------------------
@st.cache_resource
def load_model():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.load_weights("phase1_weights.weights.h5")

    return model


model = load_model()

class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal image to predict DR stage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
