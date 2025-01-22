# Import necessary libraries
import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import kagglehub
import random

# Set random seeds for reproducibility
# Setting the same seed value across libraries ensures that random operations produce consistent results.
# This is crucial in machine learning workflows to allow experiments to be replicated and results validated.
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# === Dataset Preparation ===
# Function to list all valid image files in a directory
def list_valid_images(directory):
    """
    List all image files (JPG, JPEG, PNG) in the specified directory.
    """
    return [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Function to copy images from source to destination directories
# Copying images to a single directory simplifies data loading by centralizing all files in one location.
# This avoids potential issues with scattered files and facilitates easier management of the dataset.
def copy_images(images, source_folder, destination_folder):
    """
    Copy the image files from the source folder to the destination folder.
    """
    for img in images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(destination_folder, img))

# Load the dataset from Kaggle using kagglehub
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
tumor_path = os.path.join(path, 'yes')  # Path containing images of MRI scans with tumor
no_tumor_path = os.path.join(path, 'no')  # Path containing images of MRI scans without tumor
total_path = os.path.join(path, 'Total Images')  # Combined folder for all images

# Create a directory to store all images if it doesn't exist
os.makedirs(total_path, exist_ok=True)

# List all tumor and non-tumor images
tumor_images = list_valid_images(tumor_path)
no_tumor_images = list_valid_images(no_tumor_path)

# Copy all images to a single directory to simplify access during model training
copy_images(tumor_images, tumor_path, total_path)
copy_images(no_tumor_images, no_tumor_path, total_path)

# Assign labels to the images: 1 for tumor, 0 for no tumor
labels = [1] * len(tumor_images) + [0] * len(no_tumor_images)

# Split the data into training, validation, and test sets
# Stratification ensures each set has the same distribution of labels as the original dataset.
train_images, temp_images, train_labels, temp_labels = train_test_split(
    tumor_images + no_tumor_images, labels, test_size=0.2, stratify=labels
)
validation_images, test_images, validation_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, stratify=temp_labels
)

# === Data Generator ===
# Custom data generator to preprocess and optionally augment images
class ImageDataGeneratorWithAugmentation(Sequence):
    """
    Custom data generator for handling image preprocessing and augmentation.
    """
    def __init__(self, image_paths, labels, base_path, batch_size=32, target_size=(224, 224), shuffle=True, augment=False):
        """
        Initialize the data generator.
        - image_paths: List of image file paths
        - labels: Corresponding labels for the images
        - base_path: Directory containing images
        - batch_size: Number of samples per batch
        - target_size: Desired image dimensions
        - shuffle: Whether to shuffle the data after each epoch
        - augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.base_path = base_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        """Calculate the number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle the data after each epoch if required."""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

    def __getitem__(self, index):
        """
        Generate a batch of data for training or validation.
        """
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in batch_indexes]
        batch_labels = [self.labels[k] for k in batch_indexes]

        # Preprocess the images for the current batch
        images = np.array([
            self.preprocess_image(os.path.join(self.base_path, img_path)) for img_path in batch_image_paths
        ])
        return images, np.array(batch_labels, dtype=np.float32)

    def preprocess_image(self, image_path):
        """
        Load and preprocess the image (resize and normalization).
        Optionally apply augmentation during training.
        """
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros((*self.target_size, 3))
        image = cv2.resize(image, self.target_size)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        if self.augment:
            image = self.augment_image(image)
        return image

    def augment_image(self, image):
        """
        Apply data augmentation techniques to improve model generalization.
        """
        image = tf.image.random_flip_left_right(image)  # Randomly flip horizontally
        image = tf.image.random_flip_up_down(image)    # Randomly flip vertically
        image = tf.image.random_brightness(image, max_delta=0.2)  # Randomly adjust brightness
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Randomly adjust contrast
        return image

# Instantiate data generators for training and validation
train_generator = ImageDataGeneratorWithAugmentation(
    train_images, train_labels, total_path, batch_size=32, augment=True
)
validation_generator = ImageDataGeneratorWithAugmentation(
    validation_images, validation_labels, total_path, batch_size=32, augment=False
)

# === Model Definition ===
# Load the EfficientNetB0 model with pretrained weights (excluding top layers)
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # Freeze the base model layers during training

# Define the full model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.5),  # Dropout layer for regularization
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification output layer
])

# Define the focal loss function to handle class imbalance
# Focal loss down-weights easy examples and focuses on hard-to-classify ones, addressing class imbalance.
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss function to handle class imbalance.
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * (1 - p_t) ** gamma * tf.math.log(p_t))
    return loss

# Compile the model with Adam optimizer and the defined focal loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# === Training ===
# Define callbacks for early stopping and model checkpointing to avoid overfitting and save the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # Stop early if validation loss doesn't improve
    ModelCheckpoint("best_efficientnet_model.keras", monitor="val_accuracy", save_best_only=True)
]

# Train the model with the defined generators
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,  # Train for 30 epochs
    callbacks=callbacks
)

# === Evaluation ===
# Evaluate the model on the validation set and print metrics
val_loss, val_accuracy, val_precision, val_recall, val_auc = model.evaluate(validation_generator, verbose=0)
print(f"\n=== Validation Metrics ===")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation AUC: {val_auc:.4f}")

# Generate predictions and classification report for the validation set
predictions = model.predict(validation_generator)
adjusted_val_predictions = (predictions > 0.5).astype(int)
print("\n=== Classification Report (Validation Set) ===")
print(classification_report(validation_labels, adjusted_val_predictions))

# Evaluate on the test set
test_images_preprocessed = np.array([
    validation_generator.preprocess_image(os.path.join(total_path, img)) for img in test_images
])
test_labels_preprocessed = np.array(test_labels)

# Print test evaluation metrics
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    test_images_preprocessed, test_labels_preprocessed, verbose=1
)
print(f"\n=== Test Metrics ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Generate predictions and classification report for the test set
test_predictions = model.predict(test_images_preprocessed)
adjusted_test_predictions = (test_predictions > 0.5).astype(int)
print("\n=== Classification Report (Test Set) ===")
print(classification_report(test_labels_preprocessed, adjusted_test_predictions))

# Plot confusion matrix for test set
test_conf_matrix = confusion_matrix(test_labels_preprocessed, adjusted_test_predictions)
disp_test = ConfusionMatrixDisplay(test_conf_matrix, display_labels=["No Tumor", "Tumor"])
disp_test.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Set")
plt.show()







