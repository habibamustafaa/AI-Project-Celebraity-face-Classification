import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------ Page Config ------------------
st.set_page_config(page_title="Celebrity Face Classification", layout="centered")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("resnet50_final_model.keras")

model = load_model()

# ------------------ Class Names ------------------
class_names = [
    "Angelina Jolie", "Brad Pitt", "Denzel Washington", "Hugh Jackman",
    "Jennifer Lawrence", "Johnny Depp", "Kate Winslet", "Leonardo DiCaprio",
    "Megan Fox", "Natalie Portman", "Nicole Kidman", "Robert Downey Jr",
    "Sandra Bullock", "Scarlett Johansson", "Tom Cruise", "Tom Hanks", "Will Smith"
]

# ------------------ Image Preprocessing ------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# ------------------ Prediction ------------------
def predict(image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    return [(class_names[i], float(preds[i])) for i in top3_idx]

# ------------------ UI ------------------
st.title("🎬 Celebrity Face Classification")
st.write("Upload an image or use your webcam to identify the celebrity face.")

input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

# ------------------ Upload Image ------------------
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        st.subheader("🔮 Predictions")
        results = predict(image)
        for name, conf in results:
            st.write(f"**{name}** — {conf*100:.2f}%")

# ------------------ Webcam ------------------
elif input_method == "Use Webcam":
    st.subheader("📸 Webcam Capture")
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", width=300)

        st.subheader("🔮 Predictions")
        results = predict(image)
        for name, conf in results:
            st.write(f"**{name}** — {conf*100:.2f}%")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Model: ResNet50 (Fine-tuned on Celebrity Faces Dataset)")
