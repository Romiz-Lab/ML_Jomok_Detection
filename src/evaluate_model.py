import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')
MODEL_PATH_H5 = os.path.join(BASE_DIR, 'models', 'jmk_detection.h5') # H5
MODEL_PATH_KERAS = os.path.join(BASE_DIR, 'models', 'jmk_detection_model.keras') # Keras


# Load model
model_h5 = load_model(MODEL_PATH_H5)
model_keras = load_model(MODEL_PATH_KERAS)

# Persiapan data testing
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
  TEST_DIR,
  target_size=(224, 224),
  batch_size=32,
  class_mode='categorical'
)

# Evaluasi model
result_h5 = model_h5.evaluate(test_generator)
result_keras = model_keras.evaluate(test_generator)

print('Loss:', result_h5[0])
print('Accuracy:', result_h5[1])
print('----------------------')
print('Loss:', result_keras[0])
print('Accuracy:', result_keras[1])