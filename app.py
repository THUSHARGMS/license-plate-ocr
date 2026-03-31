import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
import os

os.environ["OMP_NUM_THREADS"] = "1"

st.set_page_config(page_title="License Plate OCR", layout="centered")

def clean_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    return text

def read_plate(image):
    img = np.array(image)
    h, w = img.shape[:2]

    # Focus lower region (plate area)
    crop = img[int(h*0.5):h, int(w*0.2):int(w*0.8)]

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # OCR using Tesseract
    text = pytesseract.image_to_string(thresh)

    text = clean_text(text)

    if text.strip() == "":
        text = "No plate detected"

    # Draw box
    img_draw = img.copy()
    cv2.rectangle(img_draw,
                  (int(w*0.2), int(h*0.5)),
                  (int(w*0.8), h),
                  (0,255,0), 3)

    return img_draw, text


st.title("🚗 License Plate Recognition")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    if st.button("Detect Plate"):
        img, text = read_plate(image)

        st.image(img, caption="Detected Plate")
        st.success(f"Plate Number: {text}")
