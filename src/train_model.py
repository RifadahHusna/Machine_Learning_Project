import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIGURATION
# ==============================
DATASET_DIR = "src/dataset_dir/Label"   # ganti sesuai struktur dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ["Normal", "Tumor"]
MODEL_PATH = "best_model.h5"

# ==============================
# DATA GENERATOR
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# ==============================
# MODEL (CNN BASELINE – ganti MobileViT jika sudah siap)
# ==============================
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# CALLBACKS (BEST MODEL)
# ==============================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ==============================
# TRAINING
# ==============================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ==============================
# LOAD BEST MODEL
# ==============================
best_model = load_model(MODEL_PATH)

# ==============================
# EVALUATION
# ==============================
val_gen.reset()
preds = best_model.predict(val_gen)
pred_labels = (preds > 0.5).astype(int).flatten()
true_labels = val_gen.classes

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – Best Model")
plt.show()

# Classification Report
report = classification_report(
    true_labels,
    pred_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print("\n=== CLASSIFICATION REPORT (BEST MODEL) ===\n")
print(report)

# ==============================
# SAVE REPORT (for Streamlit)
# ==============================
with open("classification_report.txt", "w") as f:
    f.write(report)

print("\nTraining selesai. Best model disimpan sebagai:", MODEL_PATH)