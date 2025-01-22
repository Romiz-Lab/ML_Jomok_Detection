import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Konfigurasi
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Path dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
MODEL_PATH_H5 = os.path.join(BASE_DIR, 'models', 'jmk_detection_model.h5')
MODEL_PATH_KERAS = os.path.join(BASE_DIR, 'models', 'jmk_detection_model.keras')

# Persiapan data
train_datagen = ImageDataGenerator(
  rescale=1./255,
  rotation_range=20,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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

test_generator = test_datagen.flow_from_directory(
  VAL_DIR,
  target_size=IMG_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='categorical'
)

# Mendapatkan jumlah kelas
num_classes = len(train_generator.class_indices)
print(f"Jumlah kelas: {num_classes}")

# Membangun model MobileNetV2
base_model = MobileNetV2(
  weights='imagenet',
  include_top=False,
  input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# freeze base model layers
for layer in base_model.layers[:-10]:
  layer.trainable = False

# Kompilasi model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

# Training model
history = model.fit(
  train_generator,
  epochs=EPOCHS,
  validation_data=validation_generator,
  callbacks=[early_stopping, reduce_lr]
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Simpan model
model.save(MODEL_PATH_H5)
model.save(MODEL_PATH_KERAS)
print('Model berhasil disimpan')

#
# Visualisasi hasil training
#

# Plot Akurasi
plt.figure(figsize=(12, 4))
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

plt.show()
