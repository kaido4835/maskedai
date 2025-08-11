import kagglehub
import os
import shutil
import requests
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import urllib.request
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

download_dir = r"C:\Users\rahma\OneDrive\Desktop\MaskAIMiron"
models_dir = os.path.join(download_dir, "models")
output_dir = os.path.join(download_dir, "upper_face_dataset")
MODELS_DIR = os.path.join(download_dir, "visualizations")  # Директория для сохранения графиков

os.makedirs(download_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)  # Создаем директорию для визуализаций

IMG_SIZE = 96  # Размер изображений для модели

logging.basicConfig(level=logging.ERROR)

FORCE_REPROCESS = False
FORCE_RECOMBINE = False
GENERATE_VISUALIZATIONS = True  # Флаг для генерации визуализаций


def copy_files(src, dst):
    if os.path.isdir(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        files = os.listdir(src)
        for file in files:
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            copy_files(src_file, dst_file)
    else:
        shutil.copy2(src, dst)


def download_dataset(dataset_name, kaggle_dataset):
    dataset_dir = os.path.join(download_dir, dataset_name)

    if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
        print(f"Dataset {dataset_name} already exists at {dataset_dir}")
        return dataset_dir

    print(f"Downloading {dataset_name}...")

    path = kagglehub.dataset_download(kaggle_dataset)

    print(f"Original path to {dataset_name} files: {path}")

    os.makedirs(dataset_dir, exist_ok=True)

    copy_files(path, dataset_dir)

    print(f"Files copied to: {dataset_dir}")
    return dataset_dir


def download_yolo_model():
    model_path = os.path.join(models_dir, "yolov8n-face-lindevs.pt")

    if os.path.exists(model_path):
        print(f"YOLOv8 face detection model already exists at {model_path}")
        return model_path

    print("Downloading YOLOv8 face detection model...")
    url = "https://github.com/lindevs/yolov8-face/releases/download/1.0.1/yolov8n-face-lindevs.pt"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Downloading from {url}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(model_path, 'wb') as f, tqdm(
            desc="Downloading YOLOv8 model",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            f.write(data)
            progress_bar.update(len(data))

    print(f"YOLOv8 model saved to: {model_path}")
    return model_path


def download_facial_landmark_model():
    face_landmark_path = os.path.join(models_dir, "shape_predictor_68_face_landmarks.dat")

    if os.path.exists(face_landmark_path):
        print(f"Facial landmark model already exists at {face_landmark_path}")
        return face_landmark_path

    print("Downloading facial landmark model...")
    url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"

    os.makedirs(os.path.dirname(face_landmark_path), exist_ok=True)

    print(f"Downloading from {url}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(face_landmark_path, 'wb') as f, tqdm(
            desc="Downloading facial landmark model",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            f.write(data)
            progress_bar.update(len(data))

    print(f"Facial landmark model saved to: {face_landmark_path}")
    return face_landmark_path


def output_already_processed(output_subdir_path):
    if not os.path.exists(output_subdir_path):
        return False

    total_images = 0
    for root, _, files in os.walk(output_subdir_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                total_images += 1

    if total_images > 0:
        print(f"Found {total_images} already processed images in {output_subdir_path}")
        return True

    return False


def analyze_dataset_structure(dataset_dir):
    print(f"\nAnalyzing dataset structure for: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}")
        return

    extension_counts = {}
    total_images = 0

    dir_structure = {}

    for root, dirs, files in os.walk(dataset_dir):
        rel_path = os.path.relpath(root, dataset_dir)
        if rel_path == '.':
            rel_path = 'ROOT'

        dir_files = {}
        for file in files:
            _, ext = os.path.splitext(file)
            ext = ext.lower()

            if ext not in extension_counts:
                extension_counts[ext] = 0
            extension_counts[ext] += 1

            if ext not in dir_files:
                dir_files[ext] = 0
            dir_files[ext] += 1

            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                total_images += 1

        dir_structure[rel_path] = {
            'dirs': dirs,
            'file_count': len(files),
            'extensions': dir_files
        }

    print(f"Total directories: {len(dir_structure)}")
    print(f"Total image files: {total_images}")
    print("\nFile extensions found:")
    for ext, count in sorted(extension_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext}: {count} files")

    print("\nTop-level directories:")
    for dirname, content in dir_structure.items():
        if dirname == 'ROOT' or '/' not in dirname:
            print(f"  {dirname}: {content['file_count']} files, {len(content['dirs'])} subdirectories")
            if content['extensions']:
                for ext, count in sorted(content['extensions'].items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        print(f"    {ext}: {count} files")

    return total_images, dir_structure


def find_all_image_files(dataset_dir):
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def extract_upper_face(image, face_box, face_detector, landmark_predictor):
    import dlib

    x1, y1, x2, y2 = face_box

    dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))

    try:
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

        if upper_face.size == 0:
            height = y2 - y1
            upper_height = int(height * 0.45)
            upper_face = image[y1:y1 + upper_height, x1:x2]

        return upper_face, True

    except Exception as e:
        try:
            height = y2 - y1
            upper_height = int(height * 0.45)
            upper_face = image[y1:y1 + upper_height, x1:x2]
            return upper_face, upper_face.size > 0
        except:
            return None, False


def process_image_batch(image_files, out_dir, yolo_model, desc_prefix, facial_landmark_path,
                        total_images_with_faces=0, total_images_without_faces=0, total_faces_detected=0):
    import dlib
    from contextlib import redirect_stdout

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(facial_landmark_path)

    progress_bar = tqdm(
        image_files,
        desc=f"Processing {desc_prefix}",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for img_path in progress_bar:
        try:
            img_path_str = str(img_path)
            img = cv2.imread(img_path_str)
            if img is None:
                total_images_without_faces += 1
                continue

            with open(os.devnull, 'w') as f, redirect_stdout(f):
                results = yolo_model(img, verbose=False)

            faces_in_image = 0
            for i, result in enumerate(results):
                boxes = result.boxes
                if len(boxes) == 0:
                    continue

                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    upper_face, success = extract_upper_face(
                        img, (x1, y1, x2, y2), face_detector, landmark_predictor)

                    if not success or upper_face is None or upper_face.size == 0:
                        continue

                    base_name = os.path.basename(img_path_str)
                    name_without_ext, _ = os.path.splitext(base_name)
                    out_filename = f"{name_without_ext}_face{j}.jpg"
                    out_path = os.path.join(out_dir, out_filename)

                    cv2.imwrite(out_path, upper_face)

                    faces_in_image += 1
                    total_faces_detected += 1

            if faces_in_image > 0:
                total_images_with_faces += 1
            else:
                total_images_without_faces += 1

            progress_bar.set_postfix({
                'faces': faces_in_image,
                'success_rate': f"{total_images_with_faces / max(1, len(image_files)) * 100:.1f}%"
            })

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            total_images_without_faces += 1

    return total_images_with_faces, total_images_without_faces, total_faces_detected


def process_dataset(dataset_dir, output_subdir, model, facial_landmark_path, force_directory_structure=False):
    output_subdir_path = os.path.join(output_dir, output_subdir)

    if not FORCE_REPROCESS and output_already_processed(output_subdir_path):
        print(f"Skipping processing for {output_subdir} - output directory already contains processed images")
        print(f"To force reprocessing, set FORCE_REPROCESS = True at the top of the script")
        return 0, 0

    os.makedirs(output_subdir_path, exist_ok=True)

    total_images, dir_structure = analyze_dataset_structure(dataset_dir)

    if total_images == 0:
        print(f"ERROR: No image files found in the dataset directory {dataset_dir}")
        print("Please check the dataset structure and ensure it contains image files.")
        return 0, 0

    from ultralytics import YOLO

    yolo_model = YOLO(model)
    yolo_model.verbose = False

    image_dirs = []
    processing_method = "unknown"

    if force_directory_structure:
        print("Принудительно используется структура директорий для датасета")
        processing_method = "directory"

        train_test_dirs = []
        for split in ["train", "test", "val", "valid"]:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.exists(split_dir) and os.path.isdir(split_dir):
                train_test_dirs.append((split_dir, split))

        if train_test_dirs:
            for split_dir, split in train_test_dirs:
                emotion_dirs = [d for d in os.listdir(split_dir)
                                if os.path.isdir(os.path.join(split_dir, d))]

                for emotion in emotion_dirs:
                    image_dirs.append((os.path.join(split_dir, emotion), split, emotion))
        else:
            emotion_dirs = [d for d in os.listdir(dataset_dir)
                            if os.path.isdir(os.path.join(dataset_dir, d))]

            for emotion in emotion_dirs:
                if emotion.lower() in ["train", "test", "val", "valid", "__pycache__"]:
                    continue

                image_dirs.append((os.path.join(dataset_dir, emotion), "all", emotion))
    else:
        train_test_structure = False
        for split in ["train", "test", "val", "valid"]:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.exists(split_dir) and os.path.isdir(split_dir):
                train_test_structure = True

                subdirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
                if subdirs:
                    for subdir in subdirs:
                        image_dirs.append((os.path.join(split_dir, subdir), split, subdir))
                else:
                    image_dirs.append((split_dir, split, "unknown"))

        if not train_test_structure:
            emotion_structure = False
            common_emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

            subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            for subdir in subdirs:
                if subdir.lower() in ["train", "test", "val", "valid"]:
                    continue

                if subdir.lower() in common_emotions or any(
                        f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        for f in os.listdir(os.path.join(dataset_dir, subdir))
                ):
                    emotion_structure = True
                    image_dirs.append((os.path.join(dataset_dir, subdir), "unknown", subdir))

    if not image_dirs:
        all_images = find_all_image_files(dataset_dir)

        if all_images:
            print(f"Using flat processing method - found {len(all_images)} images")
            processing_method = "flat"
        else:
            print("ERROR: Could not determine dataset structure and found no images.")
            return 0, 0
    else:
        print(f"Using directory-based processing method - found {len(image_dirs)} directories")
        processing_method = "directory"

    total_images_processed = 0
    total_faces_detected = 0
    total_images_with_faces = 0
    total_images_without_faces = 0

    if processing_method == "directory":
        for img_dir, split, emotion in image_dirs:
            out_emotion_dir = os.path.join(output_subdir_path, split, emotion)
            os.makedirs(out_emotion_dir, exist_ok=True)

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(Path(img_dir).glob(f"*{ext}")))
                image_files.extend(list(Path(img_dir).glob(f"*{ext.upper()}")))

            if not image_files:
                print(f"No images found in {img_dir} - checking subdirectories")
                for subdir in [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]:
                    subdir_path = os.path.join(img_dir, subdir)
                    for ext in image_extensions:
                        image_files.extend(list(Path(subdir_path).glob(f"*{ext}")))
                        image_files.extend(list(Path(subdir_path).glob(f"*{ext.upper()}")))

            total_images_processed += len(image_files)
            print(f"Processing {len(image_files)} images from {img_dir}")

            if not image_files:
                print(f"WARNING: No images found in {img_dir}")
                continue

            images_with_faces, images_without_faces, faces = process_image_batch(
                image_files, out_emotion_dir, yolo_model, f"{split}/{emotion}", facial_landmark_path)

            total_images_with_faces += images_with_faces
            total_images_without_faces += images_without_faces
            total_faces_detected += faces

    else:
        all_images = find_all_image_files(dataset_dir)
        total_images_processed = len(all_images)

        out_dir = os.path.join(output_subdir_path, "all_images")
        os.makedirs(out_dir, exist_ok=True)

        total_images_with_faces, total_images_without_faces, total_faces_detected = process_image_batch(
            all_images, out_dir, yolo_model, "all_images", facial_landmark_path)

    print(f"\nProcessing Complete for {output_subdir}:")
    print(f"Total images processed: {total_images_processed}")
    print(
        f"Images with faces detected: {total_images_with_faces} ({total_images_with_faces / max(1, total_images_processed) * 100:.1f}%)")
    print(f"Total faces detected and cropped: {total_faces_detected}")
    print(
        f"Images without faces: {total_images_without_faces} ({total_images_without_faces / max(1, total_images_processed) * 100:.1f}%)")

    return total_images_processed, total_faces_detected


def combine_datasets():
    print("\n" + "=" * 50)
    print("Combining processed datasets for model training...")

    fer_upper_dir = os.path.join(output_dir, "fer2013_upper")
    masked_fer_upper_dir = os.path.join(output_dir, "masked_fer2013_upper")

    combined_dataset_dir = os.path.join(output_dir, "combined_upper_emotions")

    if not FORCE_RECOMBINE and os.path.exists(combined_dataset_dir) and os.listdir(combined_dataset_dir):
        print(f"Combined dataset already exists at {combined_dataset_dir}")
        print(f"To force recombination, set FORCE_RECOMBINE = True at the top of the script")

        train_images = 0
        test_images = 0
        for split in ["train", "test"]:
            split_dir = os.path.join(combined_dataset_dir, split)
            if os.path.exists(split_dir):
                for emotion_dir in os.listdir(split_dir):
                    emotion_path = os.path.join(split_dir, emotion_dir)
                    if os.path.isdir(emotion_path):
                        images = [f for f in os.listdir(emotion_path)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if split == "train":
                            train_images += len(images)
                        else:
                            test_images += len(images)

        print(f"Found {train_images} training images and {test_images} test images")
        return

    common_emotions = ["angry", "happy", "neutral", "sad", "surprise"]

    os.makedirs(combined_dataset_dir, exist_ok=True)

    def copy_files_with_progress(src_dir, dst_dir, description):
        os.makedirs(dst_dir, exist_ok=True)

        files = list(Path(src_dir).glob("*.jpg")) + list(Path(src_dir).glob("*.jpeg")) + \
                list(Path(src_dir).glob("*.png"))

        if not files:
            print(f"Warning: No images found in {src_dir}")
            return 0

        with tqdm(total=len(files), desc=description) as pbar:
            for file in files:
                dst_file = os.path.join(dst_dir, file.name)
                shutil.copy2(file, dst_file)
                pbar.update(1)

        return len(files)

    def directory_exists(dir_path):
        return os.path.exists(dir_path) and os.path.isdir(dir_path)

    print("Checking and creating directory structure...")

    fer_has_train = directory_exists(os.path.join(fer_upper_dir, "train"))
    fer_has_test = directory_exists(os.path.join(fer_upper_dir, "test"))

    masked_has_train = directory_exists(os.path.join(masked_fer_upper_dir, "train"))
    masked_has_validation = directory_exists(os.path.join(masked_fer_upper_dir, "validation"))
    masked_has_test = directory_exists(os.path.join(masked_fer_upper_dir, "test"))

    print(f"FER2013: train={fer_has_train}, test={fer_has_test}")
    print(f"Masked-FER2013: train={masked_has_train}, validation={masked_has_validation}, test={masked_has_test}")

    os.makedirs(os.path.join(combined_dataset_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(combined_dataset_dir, "test"), exist_ok=True)

    for emotion in common_emotions:
        os.makedirs(os.path.join(combined_dataset_dir, "train", emotion), exist_ok=True)
        os.makedirs(os.path.join(combined_dataset_dir, "test", emotion), exist_ok=True)

    total_train_images = 0
    total_test_images = 0

    print("\nCopying files from FER2013...")
    for emotion in common_emotions:
        if fer_has_train:
            src_dir = os.path.join(fer_upper_dir, "train", emotion)
            dst_dir = os.path.join(combined_dataset_dir, "train", emotion)
            if directory_exists(src_dir):
                count = copy_files_with_progress(
                    src_dir, dst_dir, f"FER2013 train/{emotion} -> combined train/{emotion}")
                total_train_images += count
            else:
                print(f"Warning: Directory {src_dir} not found")

        if fer_has_test:
            src_dir = os.path.join(fer_upper_dir, "test", emotion)
            dst_dir = os.path.join(combined_dataset_dir, "test", emotion)
            if directory_exists(src_dir):
                count = copy_files_with_progress(
                    src_dir, dst_dir, f"FER2013 test/{emotion} -> combined test/{emotion}")
                total_test_images += count
            else:
                print(f"Warning: Directory {src_dir} not found")

    print("\nCopying files from Masked-FER2013...")
    for emotion in common_emotions:
        if masked_has_train:
            src_dir = os.path.join(masked_fer_upper_dir, "train", emotion)
            dst_dir = os.path.join(combined_dataset_dir, "train", emotion)
            if directory_exists(src_dir):
                count = copy_files_with_progress(
                    src_dir, dst_dir, f"Masked-FER2013 train/{emotion} -> combined train/{emotion}")
                total_train_images += count
            else:
                print(f"Warning: Directory {src_dir} not found")

        if masked_has_validation:
            src_dir = os.path.join(masked_fer_upper_dir, "validation", emotion)
            dst_dir = os.path.join(combined_dataset_dir, "test", emotion)
            if directory_exists(src_dir):
                count = copy_files_with_progress(
                    src_dir, dst_dir, f"Masked-FER2013 validation/{emotion} -> combined test/{emotion}")
                total_test_images += count
            else:
                print(f"Warning: Directory {src_dir} not found")

        if masked_has_test:
            src_dir = os.path.join(masked_fer_upper_dir, "test", emotion)
            dst_dir = os.path.join(combined_dataset_dir, "test", emotion)
            if directory_exists(src_dir):
                count = copy_files_with_progress(
                    src_dir, dst_dir, f"Masked-FER2013 test/{emotion} -> combined test/{emotion}")
                total_test_images += count
            else:
                print(f"Warning: Directory {src_dir} not found")

    print("\nDataset combination complete!")
    print(f"Total images copied to train: {total_train_images}")
    print(f"Total images copied to test: {total_test_images}")
    print(f"Combined dataset saved at: {combined_dataset_dir}")

    print("\nImage distribution by class:")
    for split in ["train", "test"]:
        print(f"\n{split.capitalize()}:")
        for emotion in common_emotions:
            path = os.path.join(combined_dataset_dir, split, emotion)
            if os.path.exists(path):
                image_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {emotion}: {image_count} images")


def count_images_in_directory(directory):
    if not os.path.exists(directory):
        return {}

    image_counts = {}
    total_count = 0

    for root, dirs, files in os.walk(directory):
        rel_path = os.path.relpath(root, directory)
        if rel_path == '.':
            rel_path = ''

        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        if image_files:
            image_counts[rel_path] = len(image_files)
            total_count += len(image_files)

    image_counts['__total__'] = total_count

    return image_counts


def print_image_statistics(directory, description):
    print(f"\n{description}:")
    print(f"Directory: {directory}")

    if not os.path.exists(directory):
        print("  Directory does not exist!")
        return

    counts = count_images_in_directory(directory)

    if '__total__' in counts:
        total = counts.pop('__total__')
        print(f"  Total images: {total}")

    for path, count in sorted(counts.items()):
        if path:
            print(f"  - {path}: {count} images")
        else:
            print(f"  - [Root]: {count} images")


# Добавляем новые функции для визуализации

def visualize_data_processing_pipeline(original_image_path, model_path, facial_landmark_path):
    """
    Визуализирует весь процесс обработки от оригинального изображения до обрезанной верхней части лица
    """
    import dlib
    import matplotlib.gridspec as gridspec
    from ultralytics import YOLO

    print(f"Создание визуализации процесса обработки данных...")

    # Загружаем модели
    yolo_model = YOLO(model_path)
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(facial_landmark_path)

    # Загружаем исходное изображение
    original_img = cv2.imread(original_image_path)
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Обнаружение лица с помощью YOLO
    results = yolo_model(original_img, verbose=False)

    # Создаем фигуру
    plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

    # Исходное изображение
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(original_rgb)
    ax1.set_title('Исходное изображение', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Отображаем результат работы YOLO
    ax2 = plt.subplot(gs[0, 1])
    img_with_faces = original_rgb.copy()

    face_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(img_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ax2.imshow(img_with_faces)
    ax2.set_title('Обнаружение лица (YOLO)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Отображаем лицевые ориентиры
    ax3 = plt.subplot(gs[0, 2])
    img_with_landmarks = original_rgb.copy()

    for (x1, y1, x2, y2) in face_boxes:
        dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        landmarks = landmark_predictor(original_img, dlib_rect)

        # Рисуем все 68 ориентиров
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(img_with_landmarks, (x, y), 2, (0, 255, 0), -1)

        # Рисуем линии для бровей
        for i in range(17, 26):
            cv2.line(img_with_landmarks,
                     (landmarks.part(i).x, landmarks.part(i).y),
                     (landmarks.part(i + 1).x, landmarks.part(i + 1).y),
                     (0, 0, 255), 2)

        # Рисуем контуры глаз
        for i in range(36, 41):
            cv2.line(img_with_landmarks,
                     (landmarks.part(i).x, landmarks.part(i).y),
                     (landmarks.part(i + 1).x, landmarks.part(i + 1).y),
                     (255, 0, 0), 2)
        cv2.line(img_with_landmarks,
                 (landmarks.part(41).x, landmarks.part(41).y),
                 (landmarks.part(36).x, landmarks.part(36).y),
                 (255, 0, 0), 2)

        for i in range(42, 47):
            cv2.line(img_with_landmarks,
                     (landmarks.part(i).x, landmarks.part(i).y),
                     (landmarks.part(i + 1).x, landmarks.part(i + 1).y),
                     (255, 0, 0), 2)
        cv2.line(img_with_landmarks,
                 (landmarks.part(47).x, landmarks.part(47).y),
                 (landmarks.part(42).x, landmarks.part(42).y),
                 (255, 0, 0), 2)

    ax3.imshow(img_with_landmarks)
    ax3.set_title('Лицевые ориентиры (dlib)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Процесс выделения верхней части лица
    ax4 = plt.subplot(gs[1, 0])
    img_upper_face_region = original_rgb.copy()

    for (x1, y1, x2, y2) in face_boxes:
        dlib_rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        landmarks = landmark_predictor(original_img, dlib_rect)

        # Получаем координаты бровей и глаз
        eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
        eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 48)]

        all_points = eyebrow_points + eye_points
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]

        min_x = max(0, min(x_coords) - 20)
        max_x = min(original_rgb.shape[1], max(x_coords) + 20)
        min_y = max(0, min(y_coords) - 25)
        max_y = min(original_rgb.shape[0], max(y_coords) + 15)

        # Рисуем рамку верхней части лица
        cv2.rectangle(img_upper_face_region, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    ax4.imshow(img_upper_face_region)
    ax4.set_title('Определение верхней части лица', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Обрезанная верхняя часть лица
    ax5 = plt.subplot(gs[1, 1])

    for (x1, y1, x2, y2) in face_boxes:
        upper_face, success = extract_upper_face(
            original_img, (x1, y1, x2, y2), face_detector, landmark_predictor)

        if success and upper_face is not None:
            upper_face_rgb = cv2.cvtColor(upper_face, cv2.COLOR_BGR2RGB)
            ax5.imshow(upper_face_rgb)

    ax5.set_title('Вырезанная верхняя часть лица', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # Итоговое изображение для модели (например, измененный размер)
    ax6 = plt.subplot(gs[1, 2])

    for (x1, y1, x2, y2) in face_boxes:
        upper_face, success = extract_upper_face(
            original_img, (x1, y1, x2, y2), face_detector, landmark_predictor)

        if success and upper_face is not None:
            # Изменение размера и преобразование в оттенки серого для модели
            resized_face = cv2.resize(upper_face, (IMG_SIZE, IMG_SIZE))
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)

            # Для визуализации лучше использовать цветовую карту
            ax6.imshow(gray_face, cmap='gray')

    ax6.set_title(f'Итоговое изображение для модели ({IMG_SIZE}x{IMG_SIZE})', fontsize=14, fontweight='bold')
    ax6.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'data_processing_pipeline.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_original_datasets(fer_dir, masked_fer_dir):
    """
    Визуализирует примеры изображений и распределение классов в исходных наборах данных
    """
    print(f"Создание визуализации исходных датасетов...")

    # Определяем общие эмоции в обоих датасетах
    common_emotions = ["angry", "happy", "neutral", "sad", "surprise"]

    plt.figure(figsize=(20, 15))

    # Функция для подсчета распределения эмоций в директории
    def count_emotion_distribution(dataset_dir):
        emotion_counts = {emotion: 0 for emotion in common_emotions}

        # Проверяем структуру: train/test или сразу эмоции
        has_splits = False
        for split in ["train", "test", "val", "validation"]:
            if os.path.exists(os.path.join(dataset_dir, split)):
                has_splits = True
                break

        if has_splits:
            for split in ["train", "test", "val", "validation"]:
                split_dir = os.path.join(dataset_dir, split)
                if os.path.exists(split_dir):
                    for emotion in common_emotions:
                        emotion_dir = os.path.join(split_dir, emotion)
                        if os.path.exists(emotion_dir):
                            emotion_counts[emotion] += len([f for f in os.listdir(emotion_dir)
                                                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            for emotion in common_emotions:
                emotion_dir = os.path.join(dataset_dir, emotion)
                if os.path.exists(emotion_dir):
                    emotion_counts[emotion] += len([f for f in os.listdir(emotion_dir)
                                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        return emotion_counts

    # Получаем распределение эмоций
    fer_counts = count_emotion_distribution(fer_dir)
    masked_counts = count_emotion_distribution(masked_fer_dir)

    # Гистограмма распределения эмоций
    ax1 = plt.subplot(2, 2, 1)

    x = np.arange(len(common_emotions))
    width = 0.35

    ax1.bar(x - width / 2, [fer_counts[e] for e in common_emotions], width, label='FER2013', color='#3498db', alpha=0.8)
    ax1.bar(x + width / 2, [masked_counts[e] for e in common_emotions], width, label='Masked-FER2013', color='#e74c3c',
            alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(common_emotions, rotation=45, ha='right')
    ax1.set_ylabel('Количество изображений', fontsize=12)
    ax1.set_title('Распределение эмоций в исходных данных', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над барами
    for i, v in enumerate([fer_counts[e] for e in common_emotions]):
        ax1.text(i - width / 2, v + 20, str(v), ha='center', fontsize=9)
    for i, v in enumerate([masked_counts[e] for e in common_emotions]):
        ax1.text(i + width / 2, v + 20, str(v), ha='center', fontsize=9)

    # Функция для поиска примеров изображений
    def find_image_examples(dataset_dir, emotion, num_examples=3):
        examples = []

        # Проверяем разные возможные структуры каталогов
        for split in ["train", "test", "val", "validation", ""]:
            if split:
                emotion_dir = os.path.join(dataset_dir, split, emotion)
            else:
                emotion_dir = os.path.join(dataset_dir, emotion)

            if os.path.exists(emotion_dir):
                image_files = [f for f in os.listdir(emotion_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    # Выбираем случайные примеры
                    samples = np.random.choice(image_files, min(num_examples, len(image_files)), replace=False)
                    for sample in samples:
                        examples.append(os.path.join(emotion_dir, sample))

                    if len(examples) >= num_examples:
                        break

        return examples[:num_examples]

    # Примеры изображений из FER2013
    ax2 = plt.subplot(2, 2, 3)
    ax2.axis('off')
    ax2.set_title('Примеры из FER2013', fontsize=14, fontweight='bold')

    # Создаем сетку для примеров
    grid_size = (len(common_emotions), 3)
    inner_grid = gridspec.GridSpecFromSubplotSpec(grid_size[0], grid_size[1],
                                                  subplot_spec=plt.subplot(2, 2, 3).get_subplotspec())

    for i, emotion in enumerate(common_emotions):
        examples = find_image_examples(fer_dir, emotion)
        for j, example_path in enumerate(examples):
            if j < 3:  # Ограничиваем количество примеров
                ax = plt.Subplot(plt.gcf(), inner_grid[i, j])
                try:
                    img = plt.imread(example_path)
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"{emotion}", fontsize=8)
                    ax.axis('off')
                    plt.gcf().add_subplot(ax)
                except:
                    pass

    # Примеры изображений из Masked-FER2013
    ax3 = plt.subplot(2, 2, 4)
    ax3.axis('off')
    ax3.set_title('Примеры из Masked-FER2013', fontsize=14, fontweight='bold')

    inner_grid2 = gridspec.GridSpecFromSubplotSpec(grid_size[0], grid_size[1],
                                                   subplot_spec=plt.subplot(2, 2, 4).get_subplotspec())

    for i, emotion in enumerate(common_emotions):
        examples = find_image_examples(masked_fer_dir, emotion)
        for j, example_path in enumerate(examples):
            if j < 3:  # Ограничиваем количество примеров
                ax = plt.Subplot(plt.gcf(), inner_grid2[i, j])
                try:
                    img = plt.imread(example_path)
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"{emotion}", fontsize=8)
                    ax.axis('off')
                    plt.gcf().add_subplot(ax)
                except:
                    pass

    # Информация о датасетах
    ax4 = plt.subplot(2, 2, 2)
    ax4.axis('off')

    fer_total = sum(fer_counts.values())
    masked_total = sum(masked_counts.values())

    info_text = f"""
    FER2013:
    - Всего изображений: {fer_total}
    - Разрешение: 48x48 пикселей
    - Формат: оттенки серого
    - Публичный датасет с Kaggle

    Masked-FER2013:
    - Всего изображений: {masked_total}
    - Разрешение: 48x48 пикселей
    - Формат: оттенки серого с масками
    - Искусственно созданные маски на FER2013

    Общие характеристики:
    - 5 общих классов эмоций
    - Стандартизированные лица с фронтальным ракурсом
    - Варианты с масками и без масок
    """

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'original_datasets_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_upper_face_extraction(original_dir, processed_dir, n_examples=3):
    """
    Визуализация преобразования лиц в верхнюю часть
    """
    print(f"Создание визуализации процесса извлечения верхней части лица...")

    plt.figure(figsize=(15, 12))

    common_emotions = ["angry", "happy", "neutral", "sad", "surprise"]

    # Функция для поиска пар изображений (до/после)
    def find_image_pairs(emotion):
        pairs = []

        # Ищем изображения в исходной директории
        original_images = []
        for split in ["train", "test", "val", "validation", ""]:
            if split:
                emotion_dir = os.path.join(original_dir, split, emotion)
            else:
                emotion_dir = os.path.join(original_dir, emotion)

            if os.path.exists(emotion_dir):
                image_files = [f for f in os.listdir(emotion_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    samples = np.random.choice(image_files, min(n_examples * 3, len(image_files)), replace=False)
                    for sample in samples:
                        original_images.append((os.path.join(emotion_dir, sample), sample))

        # Ищем соответствующие обработанные изображения
        for orig_path, orig_name in original_images:
            name_base = os.path.splitext(orig_name)[0]

            # Ищем в разных возможных папках
            for split in ["train", "test", "val", "validation", ""]:
                found = False
                if split:
                    emotion_dir = os.path.join(processed_dir, split, emotion)
                else:
                    emotion_dir = os.path.join(processed_dir, emotion)

                if os.path.exists(emotion_dir):
                    for proc_file in os.listdir(emotion_dir):
                        if proc_file.startswith(name_base) and proc_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            pairs.append((orig_path, os.path.join(emotion_dir, proc_file)))
                            found = True
                            break

                if found and len(pairs) >= n_examples:
                    break

            if len(pairs) >= n_examples:
                break

        return pairs[:n_examples]

    # Создаем сетку для примеров
    rows = len(common_emotions)
    cols = n_examples * 2

    for i, emotion in enumerate(common_emotions):
        pairs = find_image_pairs(emotion)

        for j, (orig_path, proc_path) in enumerate(pairs):
            # Оригинальное изображение
            plt.subplot(rows, cols, i * cols + j * 2 + 1)
            try:
                orig_img = plt.imread(orig_path)
                plt.imshow(orig_img, cmap='gray')
                plt.title(f"Оригинал ({emotion})", fontsize=9)
                plt.axis('off')
            except:
                plt.title("Ошибка загрузки", fontsize=9)
                plt.axis('off')

            # Обработанное изображение
            plt.subplot(rows, cols, i * cols + j * 2 + 2)
            try:
                proc_img = plt.imread(proc_path)
                plt.imshow(proc_img, cmap='gray')
                plt.title(f"Верхняя часть ({emotion})", fontsize=9)
                plt.axis('off')
            except:
                plt.title("Ошибка загрузки", fontsize=9)
                plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Сравнение исходных изображений и выделенной верхней части лица", fontsize=16, fontweight='bold',
                 y=1.02)
    plt.savefig(os.path.join(MODELS_DIR, 'upper_face_extraction.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_combined_dataset(combined_dir):
    """
    Визуализация итогового объединенного набора данных
    """
    print(f"Создание визуализации объединенного набора данных...")

    plt.figure(figsize=(18, 12))

    common_emotions = ["angry", "happy", "neutral", "sad", "surprise"]
    splits = ["train", "test"]

    # Подсчет распределения изображений
    distribution = {}
    for split in splits:
        distribution[split] = {}
        for emotion in common_emotions:
            emotion_dir = os.path.join(combined_dir, split, emotion)
            if os.path.exists(emotion_dir):
                count = len([f for f in os.listdir(emotion_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                distribution[split][emotion] = count

    # График распределения по классам и выборкам
    ax1 = plt.subplot(2, 2, 1)

    x = np.arange(len(common_emotions))
    width = 0.35

    ax1.bar(x - width / 2, [distribution["train"][e] for e in common_emotions], width,
            label='Тренировочная выборка', color='#3498db')
    ax1.bar(x + width / 2, [distribution["test"][e] for e in common_emotions], width,
            label='Тестовая выборка', color='#e74c3c')

    ax1.set_xticks(x)
    ax1.set_xticklabels(common_emotions, rotation=45, ha='right')
    ax1.set_ylabel('Количество изображений', fontsize=12)
    ax1.set_title('Распределение эмоций в объединенном наборе данных', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над барами
    for i, v in enumerate([distribution["train"][e] for e in common_emotions]):
        ax1.text(i - width / 2, v + 20, str(v), ha='center', fontsize=9)
    for i, v in enumerate([distribution["test"][e] for e in common_emotions]):
        ax1.text(i + width / 2, v + 20, str(v), ha='center', fontsize=9)

    # Круговые диаграммы распределения классов
    ax2 = plt.subplot(2, 2, 2)

    train_data = [distribution["train"][e] for e in common_emotions]
    train_total = sum(train_data)
    train_percentages = [count / train_total * 100 for count in train_data]

    ax2.pie(train_data, labels=common_emotions, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.tab10(np.linspace(0, 1, len(common_emotions))),
            wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
    ax2.set_title('Распределение классов в тренировочной выборке', fontsize=14, fontweight='bold')

    ax3 = plt.subplot(2, 2, 3)

    test_data = [distribution["test"][e] for e in common_emotions]
    test_total = sum(test_data)
    test_percentages = [count / test_total * 100 for count in test_data]

    ax3.pie(test_data, labels=common_emotions, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.tab10(np.linspace(0, 1, len(common_emotions))),
            wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
    ax3.set_title('Распределение классов в тестовой выборке', fontsize=14, fontweight='bold')

    # Информация о наборе данных
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    train_total = sum(train_data)
    test_total = sum(test_data)
    total = train_total + test_total

    # Соотношение обучающей и тестовой выборок
    train_ratio = train_total / total * 100
    test_ratio = test_total / total * 100

    # Рассчитываем дисбаланс классов
    train_imbalance = max(train_percentages) - min(train_percentages)
    test_imbalance = max(test_percentages) - min(test_percentages)

    info_text = f"""
    Характеристики объединенного датасета:

    Всего изображений: {total}
    • Тренировочная выборка: {train_total} ({train_ratio:.1f}%)
    • Тестовая выборка: {test_total} ({test_ratio:.1f}%)

    Дисбаланс классов:
    • В тренировочной выборке: {train_imbalance:.1f}%
    • В тестовой выборке: {test_imbalance:.1f}%

    Особенности:
    • 5 классов эмоций (angry, happy, neutral, sad, surprise)
    • Включает изображения с масками и без масок
    • Только верхняя часть лица (глаза и брови)
    • Размер изображений: стандартизирован до {IMG_SIZE}x{IMG_SIZE} пикселей
    • Формат: оттенки серого
    """

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'combined_dataset_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_class_examples(combined_dir):
    """
    Визуализирует примеры изображений для каждого класса
    """
    print(f"Создание визуализации примеров по классам...")

    plt.figure(figsize=(20, 14))

    common_emotions = ["angry", "happy", "neutral", "sad", "surprise"]
    examples_per_emotion = 8

    # Функция для поиска примеров изображений
    def find_class_examples(emotion, split="train", n_examples=examples_per_emotion):
        examples = []

        emotion_dir = os.path.join(combined_dir, split, emotion)
        if os.path.exists(emotion_dir):
            image_files = [f for f in os.listdir(emotion_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                # Выбираем случайные примеры
                samples = np.random.choice(image_files, min(n_examples, len(image_files)), replace=False)
                for sample in samples:
                    examples.append(os.path.join(emotion_dir, sample))

        return examples

    # Создаем сетку для примеров
    rows = len(common_emotions)
    cols = examples_per_emotion

    for i, emotion in enumerate(common_emotions):
        examples = find_class_examples(emotion)

        for j, example_path in enumerate(examples):
            plt.subplot(rows, cols, i * cols + j + 1)
            try:
                img = plt.imread(example_path)
                plt.imshow(img, cmap='gray')
                if j == 0:  # Только для первого изображения в ряду
                    plt.ylabel(emotion, fontsize=14, rotation=0, labelpad=50)
                plt.axis('off')
            except:
                plt.title("Ошибка загрузки", fontsize=9)
                plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Примеры изображений по классам эмоций (верхняя часть лица)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(MODELS_DIR, 'class_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_preprocessing_statistics(original_counts, processed_counts):
    """
    Визуализирует статистику процесса обработки данных
    """
    print(f"Создание визуализации статистики предобработки данных...")

    plt.figure(figsize=(15, 10))

    # Сравнение количества изображений до и после обработки
    ax1 = plt.subplot(2, 2, 1)

    datasets = ["FER2013", "Masked-FER2013", "Объединенный"]
    original_values = [original_counts["fer"], original_counts["masked_fer"],
                       original_counts["fer"] + original_counts["masked_fer"]]
    processed_values = [processed_counts["fer_upper"], processed_counts["masked_fer_upper"],
                        processed_counts["combined"]]

    x = np.arange(len(datasets))
    width = 0.35

    ax1.bar(x - width / 2, original_values, width, label='Исходные данные', color='#3498db')
    ax1.bar(x + width / 2, processed_values, width, label='После обработки', color='#2ecc71')

    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel('Количество изображений', fontsize=12)
    ax1.set_title('Количество изображений до и после обработки', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над барами
    for i, v in enumerate(original_values):
        ax1.text(i - width / 2, v + 100, str(v), ha='center', fontsize=10)
    for i, v in enumerate(processed_values):
        ax1.text(i + width / 2, v + 100, str(v), ha='center', fontsize=10)

    # Процент сохраненных изображений
    ax2 = plt.subplot(2, 2, 2)

    retention_rates = [processed_values[i] / original_values[i] * 100 if original_values[i] > 0 else 0
                       for i in range(len(datasets))]

    bars = ax2.bar(datasets, retention_rates, color=['#3498db', '#2ecc71', '#e74c3c'])

    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Процент сохраненных изображений', fontsize=12)
    ax2.set_title('Эффективность обработки данных', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавляем значения над барами
    for i, v in enumerate(retention_rates):
        ax2.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)

    # Распределение по split
    ax3 = plt.subplot(2, 2, 3)

    splits = ["train", "test/validation"]
    if "train_test_split" in processed_counts:
        train_test_values = [processed_counts["train_test_split"]["train"],
                             processed_counts["train_test_split"]["test"]]

        ax3.pie(train_test_values, labels=splits, autopct='%1.1f%%', startangle=90,
                colors=['#3498db', '#e74c3c'],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1.5})
        ax3.set_title('Распределение тренировочного и тестового наборов', fontsize=14, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, "Информация о распределении\nтренировочного и тестового\nнаборов отсутствует",
                 ha='center', va='center', fontsize=12)
        ax3.axis('off')

    # Текстовая сводка процесса
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    info_text = f"""
    Итоги процесса подготовки данных:

    1. Исходные наборы данных:
       • FER2013: {original_counts["fer"]} изображений
       • Masked-FER2013: {original_counts["masked_fer"]} изображений

    2. После обработки:
       • Извлечено верхних частей лица из FER2013: {processed_counts["fer_upper"]} ({processed_values[0] / original_values[0] * 100:.1f}%)
       • Извлечено верхних частей лица из Masked-FER2013: {processed_counts["masked_fer_upper"]} ({processed_values[1] / original_values[1] * 100:.1f}%)

    3. Объединенный набор данных:
       • Всего изображений: {processed_counts["combined"]}
       • Покрывает 5 базовых эмоций
       • Стандартизированный размер {IMG_SIZE}x{IMG_SIZE}

    Процесс включал:
    • Обнаружение лица с помощью YOLOv8
    • Определение лицевых ориентиров (dlib)
    • Извлечение верхней части лица (глаза и брови)
    • Стандартизация размера и предобработка
    """

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'preprocessing_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_data_visualizations():
    """
    Создает все визуализации данных
    """
    print("\n" + "=" * 50)
    print("Создание визуализаций для данных...")

    # Создаем директорию для графиков если она еще не существует
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Примерные пути для визуализации обработки (нужно выбрать изображение)
    sample_original_path = None
    for emotion in ["angry", "happy", "neutral", "sad", "surprise"]:
        try:
            sample_dirs = [
                os.path.join(fer_dir, "train", emotion),
                os.path.join(fer_dir, emotion),
                os.path.join(masked_fer_dir, "train", emotion),
                os.path.join(masked_fer_dir, emotion)
            ]

            for sample_dir in sample_dirs:
                if os.path.exists(sample_dir):
                    for file in os.listdir(sample_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            sample_original_path = os.path.join(sample_dir, file)
                            break

                    if sample_original_path:
                        break

            if sample_original_path:
                break
        except:
            continue

    # Статистика наборов данных
    original_counts = {
        "fer": count_images_in_directory(fer_dir).get('__total__', 0),
        "masked_fer": count_images_in_directory(masked_fer_dir).get('__total__', 0)
    }

    processed_counts = {
        "fer_upper": count_images_in_directory(os.path.join(output_dir, "fer2013_upper")).get('__total__', 0),
        "masked_fer_upper": count_images_in_directory(os.path.join(output_dir, "masked_fer2013_upper")).get('__total__', 0),
        "combined": count_images_in_directory(os.path.join(output_dir, "combined_upper_emotions")).get('__total__', 0),
    }

    # Подсчет изображений по разделам
    train_test_split = {}
    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, "combined_upper_emotions", split)
        if os.path.exists(split_dir):
            split_count = 0
            for emotion_dir in os.listdir(split_dir):
                emotion_path = os.path.join(split_dir, emotion_dir)
                if os.path.isdir(emotion_path):
                    split_count += len([f for f in os.listdir(emotion_path)
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            train_test_split[split] = split_count

    if train_test_split:
        processed_counts["train_test_split"] = train_test_split

    # Убираем визуализацию процесса обработки, так как она не работает корректно
    # с изображениями низкого разрешения из FER2013
    # if sample_original_path:
    #    try:
    #        visualize_data_processing_pipeline(sample_original_path, model_path, facial_landmark_path)
    #    except Exception as e:
    #        print(f"Ошибка при создании визуализации процесса обработки: {str(e)}")

    try:
        visualize_original_datasets(fer_dir, masked_fer_dir)
    except Exception as e:
        print(f"Ошибка при создании визуализации исходных датасетов: {str(e)}")

    try:
        visualize_upper_face_extraction(fer_dir, os.path.join(output_dir, "fer2013_upper"))
    except Exception as e:
        print(f"Ошибка при создании визуализации извлечения верхней части лица: {str(e)}")

    try:
        visualize_combined_dataset(os.path.join(output_dir, "combined_upper_emotions"))
    except Exception as e:
        print(f"Ошибка при создании визуализации объединенного датасета: {str(e)}")

    try:
        visualize_class_examples(os.path.join(output_dir, "combined_upper_emotions"))
    except Exception as e:
        print(f"Ошибка при создании визуализации примеров по классам: {str(e)}")

    try:
        visualize_preprocessing_statistics(original_counts, processed_counts)
    except Exception as e:
        print(f"Ошибка при создании визуализации статистики предобработки: {str(e)}")

    print(f"Визуализации созданы и сохранены в {MODELS_DIR}")


# Основной блок скрипта
print("Starting dataset preparation...")

fer_dir = download_dataset("fer2013", "msambare/fer2013")
masked_fer_dir = download_dataset("masked-fer2013", "shubhanjaypandey/masked-fer2013")

masked_fer_subdir = os.path.join(masked_fer_dir, "Masked-fer2013")
if os.path.exists(masked_fer_subdir) and os.path.isdir(masked_fer_subdir):
    print(f"Using proper subdirectory for masked-fer2013: {masked_fer_subdir}")
    masked_fer_dir = masked_fer_subdir
else:
    print("WARNING: Expected subdirectory structure not found for masked-fer2013")

print_image_statistics(fer_dir, "Original FER2013 dataset")
print_image_statistics(masked_fer_dir, "Original Masked-FER2013 dataset")

model_path = download_yolo_model()

try:
    import ultralytics
except ImportError:
    print("Installing ultralytics package...")
    os.system("pip install ultralytics")
    print("Ultralytics installed.")

try:
    import dlib
except ImportError:
    print("Installing dlib package...")
    os.system("pip install dlib")
    print("dlib installed.")

facial_landmark_path = download_facial_landmark_model()

os.environ["ULTRALYTICS_HIDE_BANNER"] = "1"
os.environ["YOLO_VERBOSE"] = "0"

print("\n" + "=" * 50)
print("Processing FER2013 dataset...")
fer_images, fer_faces = process_dataset(fer_dir, "fer2013_upper", model_path, facial_landmark_path)
print_image_statistics(os.path.join(output_dir, "fer2013_upper"), "Processed FER2013 upper faces")

print("\n" + "=" * 50)
print("Processing Masked FER2013 dataset...")
masked_images, masked_faces = process_dataset(masked_fer_dir, "masked_fer2013_upper", model_path, facial_landmark_path,
                                              force_directory_structure=True)
print_image_statistics(os.path.join(output_dir, "masked_fer2013_upper"), "Processed Masked-FER2013 upper faces")

combine_datasets()

print("\n" + "=" * 50)
print("FINAL STATISTICS:")
print("=" * 50)
print_image_statistics(fer_dir, "Original FER2013 dataset")
print_image_statistics(masked_fer_dir, "Original Masked-FER2013 dataset")
print_image_statistics(os.path.join(output_dir, "fer2013_upper"), "Processed FER2013 upper faces")
print_image_statistics(os.path.join(output_dir, "masked_fer2013_upper"), "Processed Masked-FER2013 upper faces")
print_image_statistics(os.path.join(output_dir, "combined_upper_emotions"), "Combined dataset with 5 emotions")

# После завершения обработки данных и перед выводом заключительной статистики
if GENERATE_VISUALIZATIONS:
    create_data_visualizations()

print("\n" + "=" * 50)
print("All processing complete!")

if fer_images > 0 or masked_images > 0:
    print(f"Total images processed: {fer_images + masked_images}")
    print(f"Total faces detected and cropped: {fer_faces + masked_faces}")

print(f"Dataset location: {output_dir}")
print(f"Visualizations location: {MODELS_DIR}")
print("=" * 50)