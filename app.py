import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Define sub-categories and mapping to main categories
SUB_CATEGORIES = [
    'Broken Guard Rails',
    'Damaged Signs',
    'Illegal Parking',
    'Vandalism / Graffiti',
]
MAIN_CATEGORY_MAP = {
    'Broken Guard Rails': 'Public Safety',
    'Damaged Signs': 'Road Issues',
    'Illegal Parking': 'Road Issues',
    'Vandalism / Graffiti': 'Public Cleanliness',
}

# Load trained model
model = load_model('final_model.h5')

st.title("Urban Issue Classification Demo")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    threshold = 0.5  # You can adjust this threshold
    detected_subs = [SUB_CATEGORIES[i] for i, p in enumerate(preds) if p > threshold]
    detected_mains = sorted(set([MAIN_CATEGORY_MAP[sub] for sub in detected_subs]))

    st.write("### Predicted Main Categories:")
    st.write(detected_mains if detected_mains else "None detected")
    st.write("### Predicted Sub Categories:")
    st.write(detected_subs if detected_subs else "None detected")