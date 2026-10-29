
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "ZariraRahman/Skin-Cancer-Detection-app"

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬"
)

st.title("Skin Cancer Detection CNN")
st.write("Upload a skin image for an experimental model prediction.")

st.warning(
    "This application is for educational purposes only. "
    "It cannot diagnose skin cancer. Consult a qualified healthcare "
    "professional about any concerning skin changes."
)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="skin_cancer_cnn.keras"
    )
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

@st.cache_data
def load_labels():
    labels_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="class_names.json"
    )
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        labels = [
            labels[k] for k in sorted(
                labels, key=lambda x: int(x)
                if x.isdigit() else x
            )
        ]
    return labels

try:
    model = load_model()
    labels = load_labels()
except Exception as e:
    st.error("Model load nahi hua. Repository files check karein.")
    st.exception(e)
    st.stop()

uploaded = st.file_uploader(
    "Skin image upload karein",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image",
             use_container_width=True)

    if st.button("Predict"):
        try:
            shape = model.input_shape
            height = shape[1] or 224
            width = shape[2] or 224

            resized = image.resize((width, height))
            array = np.asarray(resized, dtype=np.float32)
            array = np.expand_dims(array, axis=0)

            output = np.asarray(
                model.predict(array, verbose=0)
            ).reshape(-1)

            if len(output) != len(labels) or len(output) < 2:
                st.error(
                    "Model output aur class labels match nahi karte. "
                    "Model configuration verify karni hogi."
                )
            else:
                index = int(np.argmax(output))
                st.subheader("Model result")
                st.write("Predicted class:", labels[index])
                st.write(
                    "Model score:",
                    f"{float(output[index]) * 100:.2f}%"
                )
                st.caption(
                    "Score confirmed diagnosis ya reliable medical "
                    "probability nahi hai."
                )
        except Exception as e:
            st.error(f"Prediction error: {e}")
