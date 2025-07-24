import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown

st.title("Hello")
#model_file = 'cat_dog_model.h5'

# Download from Google Drive if not exists
#file_id = '1fUtsl5VJv_F6lz-l2IaGaQxyGVDpOqAm'  # Replace with your actual file ID
#url = f'https://drive.google.com/uc?id={file_id}'

#if not os.path.exists(model_file):
#    st.info("Downloading model from Google Drive...")
#    gdown.download(url, model_file, quiet=False)

# Load the model
#@st.cache_resource
#def load_my_model():
#    model = tf.keras.models.load_model(model_file)
#    return model

#model = load_my_model()

#st.success("Model loaded successfully!")
#st.write(model.summary())




# Load model
#model = tf.keras.models.load_model("cat_dog_model.h5")


# Preprocessing
#def preprocess_image(img):
#    img = img.resize((150, 150))  # resize for model input
#    img_array = np.array(img) / 255.0
#    return img_array.reshape(1, 150, 150, 3)


# UI
#st.set_page_config(page_title="Cat vs Dog Classifier")
#st.title("🐱🐶 Cat or Dog? - Image Classifier")
#st.write("Upload an image to check if it's a cat or a dog!")

#uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#if uploaded_file is not None:
#    image = Image.open(uploaded_file)
#    st.image(image, caption="Uploaded Image", use_column_width=True)

#    if st.button("Predict"):
#        img_preprocessed = preprocess_image(image)
#        prediction = model.predict(img_preprocessed)[0][0]

#        label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
#        st.subheader(f"Prediction: **{label}**")
#        st.write(f"Confidence: **{prediction:.2f}** (0 = Cat, 1 = Dog)")
