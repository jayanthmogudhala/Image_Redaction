import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import pytesseract
import cv2
import os
from werkzeug.utils import secure_filename
import re

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

# Regex Patterns for sensitive data
sensitive_patterns = {
    'Aadhaar Number': r'\b\d{4}\s\d{4}\s\d{4}\b',
    'PAN Card Number': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
    'Date of Birth': r'\b\d{2}/\d{2}/\d{4}\b',
    'Name': r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',
    'Address': r'(Road|Street|St|Rd|Avenue|Pashan|Block|Hotel|Residency|House|Row|Near|Pune)',
}

# Helper functions (same as in the Flask code)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def detect_sensitive_text(image_path):
    image = Image.open(image_path)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    sensitive_boxes = []
    for i, text in enumerate(ocr_data['text']):
        for label, pattern in sensitive_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                sensitive_boxes.append((x, y, w, h))
    return sensitive_boxes

def detect_faces(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return faces

def get_redaction_boundary(level, face_x, face_y, face_w, face_h):
    if level == 'Low':
        return (face_x, face_y, face_x + face_w, face_y + face_h)
    elif level == 'Moderate':
        return (face_x - int(face_w * 0.3), face_y - int(face_h * 0.4), face_x + face_w + int(face_w * 0.3), face_y + face_h + int(face_h * 0.5))
    elif level == 'High':
        return (face_x - int(face_w * 0.5), face_y - int(face_h * 0.5), face_x + face_w + int(face_w * 0.5), face_y + face_h + int(face_h * 0.7))
    else:
        raise ValueError("Invalid level. Please choose 'Low', 'Moderate', or 'High'.")

def redact_regions(image_path, faces, sensitive_text_boxes, output_path, blur_radius=15):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for (x, y, w, h) in faces:
        redaction_boundary = get_redaction_boundary('High', x, y, w, h)  # For now using 'High' for face redaction
        face_region = image.crop(redaction_boundary)
        blurred_face = face_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_face, redaction_boundary[:2])

    for (x, y, w, h) in sensitive_text_boxes:
        region = image.crop((x, y, x + w, y + h))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        image.paste(blurred_region, (x, y))

    image.save(output_path)
    return output_path

# Streamlit UI and Backend
st.title("Image Redaction Tool")
uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join("uploads", secure_filename(uploaded_file.name))
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, caption="Uploaded Image", use_container_width=True)

    # Slider for blur radius percentage (0 to 100%)
    blur_percentage = st.slider("Select Blur Percentage", 0, 100, 50)  # Default set to 50%
    blur_radius = blur_percentage * 0.2  # Convert percentage to a blur radius factor

    # Perform Redaction when button is pressed
    if st.button("Redact Image"):
        sensitive_text_boxes = detect_sensitive_text(file_path)
        faces = detect_faces(file_path)

        output_filename = f"redacted_{uploaded_file.name}"
        output_path = os.path.join("uploads", output_filename)

        # Redact the image
        redacted_image_path = redact_regions(file_path, faces, sensitive_text_boxes, output_path=output_path, blur_radius=blur_radius)

        # Display the redacted image
        st.image(redacted_image_path, caption="Redacted Image", use_column_width=True)

        # Provide download button
        with open(redacted_image_path, "rb") as f:
            st.download_button(
                label="Download Redacted Image",
                data=f,
                file_name=output_filename,
                mime="image/png"
            )
