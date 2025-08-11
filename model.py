import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

DATA_PATH = "C:/Users/rahma/OneDrive/Desktop/MaskAIMiron/upper_face_dataset/combined_upper_emotions"
TRAIN_DIR = os.path.join(DATA_PATH, "train")
TEST_DIR = os.path.join(DATA_PATH, "test")
MODELS_DIR = "trained_models"

os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5
LEARNING_RATE = 0.001

def check_internet_connection():
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("Internet connection detected. Pre-trained weights will be loaded.")
        return True
    except OSError:
        print("Internet connection not detected. Working in offline mode.")
        return False

OFFLINE_MODE = not check_internet_connection()

def create_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

def create_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    try:
        if OFFLINE_MODE:
            print("Initializing MobileNetV2 without pre-trained weights (offline mode)...")
            base_model = MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=input_shape
            )
            weights_loaded = False
        else:
            print("Loading pre-trained MobileNetV2 weights...")
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            print("MobileNetV2 weights successfully loaded!")
            weights_loaded = True
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        print("Automatically switching to offline mode for MobileNetV2...")
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        weights_loaded = False

    if weights_loaded:
        print("Freezing initial layers for transfer learning...")
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

def create_efficientnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    try:
        if OFFLINE_MODE:
            print("Initializing EfficientNetB0 without pre-trained weights (offline mode)...")
            base_model = EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=input_shape
            )
            weights_loaded = False
        else:
            print("Loading pre-trained EfficientNetB0 weights...")
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
            print("EfficientNetB0 weights successfully loaded!")
            weights_loaded = True
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        print("Automatically switching to offline mode for EfficientNetB0...")
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=input_shape
        )
        weights_loaded = False

    if weights_loaded:
        print("Freezing initial layers for transfer learning...")
        for layer in base_model.layers[:-15]:
            layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

def create_data_generators(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def get_class_weights(train_dir):
    class_counts = {
        'angry': 1330,
        'happy': 3392,
        'neutral': 1954,
        'sad': 1325,
        'surprise': 1342
    }

    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)

    class_weights = {}
    for i, (class_name, count) in enumerate(sorted(class_counts.items())):
        class_weights[i] = (total_samples / (n_classes * count))

    return class_weights

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title(f'Model accuracy {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title(f'Model loss {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'{model_name}_training_history.png'))
    plt.show()

def evaluate_model(model, test_generator, model_name, class_names):
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)

    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"Classification report for {model_name}:")
    for cls in class_names:
        print(
            f"{cls}: Precision={report[cls]['precision']:.4f}, Recall={report[cls]['recall']:.4f}, F1-score={report[cls]['f1-score']:.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.show()

    with open(os.path.join(MODELS_DIR, f'{model_name}_report.txt'), 'w') as f:
        f.write(f"Classification report for {model_name}:\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n\n")
        for cls in class_names:
            f.write(
                f"{cls}: Precision={report[cls]['precision']:.4f}, Recall={report[cls]['recall']:.4f}, F1-score={report[cls]['f1-score']:.4f}\n")

    return report['accuracy']

def train_and_evaluate_model(model_builder, model_name, epochs=EPOCHS):
    model_best_path = os.path.join(MODELS_DIR, f'{model_name}_best.h5')
    model_full_path = os.path.join(MODELS_DIR, f'{model_name}_full.h5')

    if os.path.exists(model_full_path):
        print(f"Existing model {model_name} found. Loading model without retraining...")
        model = load_model(model_full_path)

        _, _, test_generator = create_data_generators(TRAIN_DIR, TEST_DIR)
        class_names = sorted(test_generator.class_indices.keys())

        accuracy = evaluate_model(model, test_generator, model_name, class_names)

        return model, accuracy

    print(f"Training new model {model_name}...")

    train_generator, validation_generator, test_generator = create_data_generators(TRAIN_DIR, TEST_DIR)
    class_names = sorted(train_generator.class_indices.keys())

    class_weights = get_class_weights(TRAIN_DIR)

    model = model_builder()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(
            model_best_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights
        )

        model.load_weights(model_best_path)
        accuracy = evaluate_model(model, test_generator, model_name, class_names)

        plot_training_history(history, model_name)

        model.save(model_full_path)

        return model, accuracy

    except KeyboardInterrupt:
        print(f"\nTraining of model {model_name} interrupted by user.")
        if os.path.exists(model_best_path):
            print(f"Loading last saved weights from {model_best_path}")
            model.load_weights(model_best_path)
            accuracy = evaluate_model(model, test_generator, model_name, class_names)
            return model, accuracy
        else:
            print("No saved weights. Returning untrained model.")
            return model, 0.0

def compare_models():
    models = [
        (create_custom_cnn, "CustomCNN"),
        (create_mobilenet_model, "MobileNetV2"),
        (create_efficientnet_model, "EfficientNetB0")
    ]

    results = {}

    for model_builder, model_name in models:
        print(f"\n{'=' * 50}")
        print(f"Training and evaluating model: {model_name}")
        print(f"{'=' * 50}")

        model, accuracy = train_and_evaluate_model(model_builder, model_name)
        results[model_name] = accuracy

    print("\nFinal model comparison:")
    for model_name, accuracy in results.items():
        print(f"{model_name}: Accuracy = {accuracy:.4f}")

    if results:
        best_model = max(results, key=results.get)
        print(f"\nBest model: {best_model} with accuracy {results[best_model]:.4f}")

def test_on_masked_faces(model_name, masked_test_dir=None):
    if masked_test_dir is None:
        print("Path to masked faces test dataset not specified.")
        return

    model_path = os.path.join(MODELS_DIR, f'{model_name}_full.h5')
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}")
        return

    print(f"Loading model {model_name}...")
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    try:
        masked_test_generator = test_datagen.flow_from_directory(
            masked_test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        class_names = sorted(masked_test_generator.class_indices.keys())

        print(f"\nTesting model {model_name} on masked faces:")
        evaluate_model(model, masked_test_generator, f"{model_name}_masked", class_names)
    except Exception as e:
        print(f"Error during testing: {str(e)}")

def visualize_activations(model_name, img_path):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model

    model_path = os.path.join(MODELS_DIR, f'{model_name}_full.h5')
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}")
        return

    print(f"Loading model {model_name}...")
    base_model = load_model(model_path)

    layer_names = []
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_names.append(layer.name)

    layer_names = layer_names[:3]

    print(f"Visualizing layer activations: {', '.join(layer_names)}")

    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        activation_models = []
        for layer_name in layer_names:
            activation_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
            activation_models.append((layer_name, activation_model))

        for layer_name, activation_model in activation_models:
            activations = activation_model.predict(x)

            n_features = min(16, activations.shape[-1])
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols

            plt.figure(figsize=(12, 12))
            for i in range(n_features):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(activations[0, :, :, i], cmap='viridis')
                plt.title(f'Channel {i}')
                plt.axis('off')

            plt.suptitle(f'Activations of {layer_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(MODELS_DIR, f'activations_{model_name}_{layer_name}.png'))
            plt.show()
    except Exception as e:
        print(f"Error visualizing activations: {str(e)}")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    print("GPU available:", tf.config.list_physical_devices('GPU'))

    os.makedirs(MODELS_DIR, exist_ok=True)

    compare_models()