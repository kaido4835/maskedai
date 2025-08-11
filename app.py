import os
import cv2
import numpy as np
import tensorflow as tf
import dlib
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback
import warnings
import logging


st.config.gather_sources_exclude = ("torch.*", "ultralytics.*", "cv2.*", "tensorflow.*")


def determine_input_size(model):
    try:
        input_shape = model.input_shape

        if input_shape is not None and len(input_shape) >= 4:
            height, width = input_shape[1], input_shape[2]
            if height is not None and width is not None:
                return (height, width)

        standard_sizes = [(48, 48), (96, 96), (224, 224)]

        for size in standard_sizes:
            try:
                test_img = np.zeros((1, size[0], size[1], 3), dtype=np.float32)
                model.predict(test_img, verbose=0)
                return size
            except Exception as e:
                continue

        return (96, 96)

    except Exception as e:
        return (96, 96)


warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
st.config.gather_sources_exclude = ("torch.*", "ultralytics.*", "cv2.*")

st.set_page_config(
    page_title="Emotion Recognition from Upper Face",
    page_icon="😊",
    layout="wide"
)

MODELS_DIR = "models"
OUTPUT_DIR = "output"
TRAINED_MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_LABELS = ["angry", "happy", "neutral", "sad", "surprise"]


@st.cache_resource
def load_models():
    models = {}

    yolo_model_path = os.path.join(MODELS_DIR, "yolov8n-face-lindevs.pt")
    if not os.path.exists(yolo_model_path):
        st.warning("YOLOv8 face detection model not found. Please download it manually.")
        st.markdown(
            "[Download YOLOv8 face detection model](https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt)")
        st.info(f"Save the model to directory: {yolo_model_path}")
    else:
        models["yolo"] = YOLO(yolo_model_path)
        models["yolo"].verbose = False

    landmark_model_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(landmark_model_path):
        st.warning("dlib facial landmarks model not found.")
        st.markdown(
            "[Download dlib facial landmarks model](https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat)")
        st.info(f"Save the model to directory: {landmark_model_path}")
    else:
        models["face_detector"] = dlib.get_frontal_face_detector()
        models["landmark_predictor"] = dlib.shape_predictor(landmark_model_path)

    emotion_model_path = os.path.join(TRAINED_MODELS_DIR, "CustomCNN_full.h5")
    if not os.path.exists(emotion_model_path):
        st.error(
            f"CustomCNN emotion recognition model not found. Make sure the file is in the directory '{TRAINED_MODELS_DIR}'")
    else:
        try:
            models["emotion"] = tf.keras.models.load_model(emotion_model_path)

            num_classes = models["emotion"].output_shape[-1]
            if num_classes != len(EMOTION_LABELS):
                st.warning(
                    f"Warning: model has {num_classes} classes, but {len(EMOTION_LABELS)} emotions are specified in the code.")
                st.warning("Make sure the EMOTION_LABELS list matches the model's classes!")

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    return models


def detect_faces(image, yolo_model):
    results = yolo_model(image, verbose=False)

    faces = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            faces.append((x1, y1, x2, y2))

    return faces, results


def extract_upper_face(image, face_box, face_detector, landmark_predictor):
    try:
        x1, y1, x2, y2 = face_box

        dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))

        landmarks = landmark_predictor(image, dlib_rect)

        eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
        eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 48)]

        all_points = eyebrow_points + eye_points
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]

        min_x = max(0, min(x_coords) - 20)
        max_x = min(image.shape[1], max(x_coords) + 20)
        min_y = max(0, min(y_coords) - 25)
        max_y = min(image.shape[0], max(y_coords) + 15)

        upper_face = image[min_y:max_y, min_x:max_x]

        return upper_face, (min_x, min_y, max_x, max_y), True

    except Exception as e:
        try:
            height = y2 - y1
            upper_height = int(height * 0.45)
            upper_face = image[y1:y1 + upper_height, x1:x2]
            return upper_face, (x1, y1, x2, y1 + upper_height), upper_face.size > 0
        except:
            return None, (0, 0, 0, 0), False


def preprocess_image(image, target_size=(96, 96)):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    gray_image = cv2.resize(gray_image, target_size)

    gray_image = gray_image.astype('float32') / 255.0

    rgb_image = np.stack([gray_image] * 3, axis=-1)

    rgb_image = np.expand_dims(rgb_image, axis=0)

    return rgb_image


def recognize_emotion(face_image, emotion_model, input_size=(96, 96), debug_mode=False):
    try:
        processed_image = preprocess_image(face_image, target_size=input_size)

        if debug_mode:
            st.sidebar.write(f"Input image: {processed_image.shape}, dtype: {processed_image.dtype}")
            input_shape = emotion_model.input_shape
            st.sidebar.write(f"Expected input shape: {input_shape}")

            debug_image = processed_image[0] * 255.0
            debug_image = debug_image.astype(np.uint8)
            cv2.imwrite('temp_processed_image.jpg', debug_image)

        prediction = emotion_model.predict(processed_image, verbose=0)[0]

        emotion_idx = np.argmax(prediction)
        emotion_label = EMOTION_LABELS[emotion_idx]

        return emotion_label, prediction

    except Exception as e:
        st.error(f"Error recognizing emotion: {str(e)}")
        st.error(traceback.format_exc())

        return "error", np.zeros(len(EMOTION_LABELS))


def visualize_results(image, face_box, upper_face_box, emotion, probabilities):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = face_box
    axs[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none'))

    uf_x1, uf_y1, uf_x2, uf_y2 = upper_face_box
    axs[0].add_patch(
        patches.Rectangle((uf_x1, uf_y1), uf_x2 - uf_x1, uf_y2 - uf_y1, linewidth=2, edgecolor='r', facecolor='none'))
    axs[0].set_title("Detected Face")
    axs[0].axis('off')

    uf_img = image[uf_y1:uf_y2, uf_x1:uf_x2]
    axs[1].imshow(cv2.cvtColor(uf_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Upper Face")
    axs[1].axis('off')

    bars = axs[2].bar(EMOTION_LABELS, probabilities * 100)

    emotion_idx = EMOTION_LABELS.index(emotion)
    bars[emotion_idx].set_color('green')

    axs[2].set_title("Emotion Probability Distribution")
    axs[2].set_ylabel("Probability (%)")
    axs[2].set_ylim(0, 100)

    for i, v in enumerate(probabilities):
        axs[2].text(i, v * 100 + 3, f"{v * 100:.1f}%", ha='center')

    plt.tight_layout()
    return fig


def process_image(image, models, debug_mode=False):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3 and isinstance(image, np.ndarray):
            if not isinstance(image, np.ndarray):
                image = np.array(image)

        faces, results = detect_faces(image, models["yolo"])

        if not faces:
            return None, "No face detected in the image", None

        face_box = faces[0]

        upper_face, upper_face_box, success = extract_upper_face(
            image, face_box, models["face_detector"], models["landmark_predictor"])

        if not success or upper_face is None or upper_face.size == 0:
            return None, "Failed to extract upper face", None

        cv2.imwrite('temp_upper_face.jpg', upper_face)

        if debug_mode:
            st.sidebar.write(f"Size of extracted upper face: {upper_face.shape}")

        input_size = models.get("input_size", (96, 96))

        emotion, probabilities = recognize_emotion(upper_face, models["emotion"], input_size, debug_mode)

        if emotion == "error":
            return None, "Error recognizing emotion", None

        fig = visualize_results(image, face_box, upper_face_box, emotion, probabilities)

        return fig, emotion, probabilities

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error(traceback.format_exc())
        return None, f"An error occurred: {str(e)}", None


def main():
    st.title("Emotion Recognition from Upper Face")

    with st.expander("About the Project"):
        st.markdown("""
        This application recognizes emotions from the upper part of the face (eyes and eyebrows area),
        which is especially useful in situations where the lower part of the face is covered (for example, by a mask).

        **Supported emotions:**
        - Angry
        - Happy
        - Neutral
        - Sad
        - Surprise

        **Technologies:**
        - YOLOv8 for face detection
        - dlib for facial landmark detection
        - CustomCNN for emotion classification
        """)

    models = load_models()

    required_models = ["yolo", "face_detector", "landmark_predictor", "emotion"]
    missing_models = [model for model in required_models if model not in models]

    if missing_models:
        st.error(
            f"The following models are missing: {', '.join(missing_models)}. Download them to use the application.")
        return

    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    source = st.radio("Select image source:", ["Upload image", "Use camera"])

    if source == "Upload image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("Recognize Emotion"):
                with st.spinner("Processing image..."):
                    fig, emotion, probabilities = process_image(image, models, debug_mode)

                    if fig is not None:
                        st.subheader(f"Recognized emotion: {emotion.upper()}")
                        st.pyplot(fig)
                    else:
                        st.error(emotion)

    else:
        picture = st.camera_input("Take a photo")

        if picture is not None:
            with st.spinner("Processing image..."):
                fig, emotion, probabilities = process_image(Image.open(picture), models, debug_mode)

                if fig is not None:
                    st.subheader(f"Recognized emotion: {emotion.upper()}")
                    st.pyplot(fig)
                else:
                    st.error(emotion)

    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Select image source (upload or camera)
    2. Make sure the face is clearly visible and well-lit
    3. Click the "Recognize Emotion" button or take a photo

    **Recommendations:**
    - Use images with good lighting
    - Make sure the face is in frame and not obscured
    - For best results, the face should be facing the camera
    """)

    st.sidebar.header("Model Accuracy")
    st.sidebar.markdown("""
    Accuracy of the CustomCNN model on the test dataset: **67.83%**

    Other tested models:
    - MobileNetV2: 60.08%
    - EfficientNetB0: 35.61%
    """)

    if debug_mode and "emotion" in models:
        st.sidebar.header("Model Information")
        st.sidebar.write(f"Input shape: {models['emotion'].input_shape}")
        st.sidebar.write(f"Output shape: {models['emotion'].output_shape}")

        st.sidebar.header("Debug Images")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if os.path.exists('temp_upper_face.jpg'):
                st.write("Upper face")
                st.image('temp_upper_face.jpg')

        with col2:
            if os.path.exists('temp_processed_image.jpg'):
                st.write("Preprocessing")
                st.image('temp_processed_image.jpg')

        with st.sidebar.expander("Model Summary"):
            model_summary_str = []
            models["emotion"].summary(print_fn=lambda x: model_summary_str.append(x))
            st.code("\n".join(model_summary_str), language="python")


if __name__ == "__main__":
    main()