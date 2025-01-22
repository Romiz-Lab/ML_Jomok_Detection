import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import subprocess

# Konfigurasi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'jmk_detection_model.keras')
TOKEN = '7249254292:AAH3367b3c_QmHyDsgQByaSh1qTSqw8Ngt0'

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model = load_model(MODEL_PATH)

# Prediksi gambar
def predict_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.

  predictions = model.predict(img_array)
  classes = ['Anime', 'Hentai', 'Jomok', 'Neutral', 'Porn', 'Sexy']

  class_confidences = {classes[i]: predictions[0][i] for i in range(len(classes))}

  result = classes[np.argmax(predictions)]
  confidence = np.max(predictions)

  return result, confidence, class_confidences

# Cek gambar
async def check_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
  if update.message.photo or update.message.sticker:
    # Ambil file gambar/stiker
    if update.message.photo:
      file = await update.message.photo[-1].get_file()
      file_path = "temp_img.jpeg"
    else: # Stiker
      file = await update.message.sticker.get_file()
      file_path = "temp_sticker.webp"

    await file.download_to_drive(file_path)

    # Prediksi kategori gambar
    result, confidence, class_confidences = predict_image(file_path)

    # Tentukan kategori yg harus dihapus
    if result in ['Jomok', 'Hentai', 'Porn'] and confidence >= 0.90:
      try:
        await update.message.delete()
        username = update.message.from_user.username
        await context.bot.send_message(
          chat_id=update.effective_chat.id,
          text=f"Pesan dari @{username} dihapus karena terindikasi **{result}** dengan akurasi **{confidence:.2f}**. "
               f"Jika bot ini salah, silakan hubungi admin @romiyusna.",
          parse_mode='Markdown'
        )
        logger.info(f"Pesan dari @{username} dihapus karena terindikasi {result} dengan akurasi {confidence:.2f}.")
      except Exception as e:
        logger.error(f"Gagal mengapus pesan: {str(e)}")
    else:
      logger.info(f"Prediksi untuk gambar/stiker dari @{update.message.from_user.username} "
                  f"adalah {result} dengan confidence {confidence:.2f}, tidak termasuk kategori. Pesan tidak dihapus.")

    os.remove(file_path)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text("Bot siap memantau gambar dan stiker!")

def main():
  application = Application.builder().token(TOKEN).build()

  application.add_handler(MessageHandler(filters.PHOTO | filters.Sticker.ALL, check_media))
  application.add_handler(CommandHandler("start", start))

  logger.info("Bot mulai berjalan...")
  application.run_polling()

if __name__ == "__main__":
  main()