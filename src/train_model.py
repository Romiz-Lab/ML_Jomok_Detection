import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Konfigurasi
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mobilenetv2_improved.keras')

# Data generator
train_datagen = ImageDataGenerator(
  rescale=1.0 / 255,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load dataset
train_generator = train_datagen.flow_from_directory(
  TRAIN_DIR,
  target_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
  VAL_DIR,
  target_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='categorical'
)

# Balancing data
class_weights = compute_class_weight(
  class_weight='balanced',
  classes=np.unique(train_generator.classes),
  y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Mendapatkan jumlah kelas
num_classes = len(train_generator.class_indices)
print(f"Jumlah kelas: {num_classes}")

# Membangun model MobileNetV2
base_model = MobileNetV2(
  weights='imagenet',
  include_top=False,
  input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Unfreeze beberapa layer terakhir untuk fine-tuning
for layer in base_model.layers[-30:]:
  layer.trainable = True

# Tambah kustom layer dengan regularisasi
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Kompilasi model
model.compile(
  optimizer=Adam(learning_rate=0.00005),
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Training model
history = model.fit(
  train_generator,
  epochs=EPOCHS,
  validation_data=validation_generator,
  class_weight=class_weights,
  callbacks=[early_stopping, reduce_lr]
)

# Evaluasi model
test_loss, test_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Simpan model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f'Model berhasil disimpan di {MODEL_PATH}')

# Visualisasi hasil training
plt.figure(figsize=(12, 4))

# Plot Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'grafik', 'training_results.png'))
plt.show()