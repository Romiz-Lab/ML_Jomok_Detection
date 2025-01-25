import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, 'models', 'jmk_detection.h5') # H5
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'efficientnet_finetuned_model.keras') # Keras

# Load model
model = load_model(MODEL_PATH)

# Fungsi prediksi gambar
def predict_image(image_path):
  img = image.load_img(image_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.

  predictions = model.predict(img_array)
  classes = ['Anime', 'Hentai', 'Jomok', 'Neutral', 'Porn', 'Sexy']

  # Hasil prediksi untuk setiap kelas
  class_confidences = {classes[i]: predictions[0][i] for i in range(len(classes))}

  # Kelas dengan akurasi tertinggi
  result = classes[np.argmax(predictions)]
  confidence = np.max(predictions)

  return result, confidence, class_confidences

# Penggunaan
if __name__ == "__main__":
  image_path = os.path.join(BASE_DIR, 'data', 'test',  'p4.jpg') # Ganti dengan path gambar yang ingin diprediksi

  result, confidence, class_confidences = predict_image(image_path)
  print("----------------------")
  print(f'Prediksi: {result}')
  print(f'Akurasi: {confidence:.2f}')
  print("----------------------")
  print('Akurasi kategori lainnya:')
  for class_name, class_confidence in class_confidences.items():
    print(f' → {class_name}: {class_confidence:.2f}')