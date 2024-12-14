import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load the trained model
model_name = "ResNet50"
model_save_path = "modelResNet50_model.pth"  # Ganti dengan path model yang sesuai

# Jumlah kelas dan nama kelas
num_classes = 9  # Ganti dengan jumlah kelas yang sesuai
classes = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']  # Ganti dengan nama kelas yang sesuai

# Memuat model ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Menentukan device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_save_path, map_location=device))  # Memuat model ke device yang sesuai
model.eval()
model.to(device)

# Definisikan transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ukuran gambar
    transforms.ToTensor(),          # Mengonversi ke tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisasi
])

# Fungsi untuk mengklasifikasikan gambar
def classify_image(image):
    try:
        # Membaca gambar dan menerapkan transformasi
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Melakukan prediksi
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        # Menentukan kelas yang diprediksi
        predicted_class = predicted.item()
        predicted_class_name = classes[predicted_class]

        # Menampilkan hasil prediksi
        st.image(image, caption=f"Prediksi Kelas: {predicted_class_name}", use_column_width=True)
        st.write(f"Prediksi kelas gambar ini adalah: {predicted_class_name}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengklasifikasikan gambar: {e}")

# UI Streamlit
st.title("Klasifikasi Gambar dengan Model ResNet50")
st.write("Upload gambar yang ingin Anda klasifikasikan menggunakan model yang telah dilatih.")

# Menambahkan opsi untuk mengunggah gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka gambar yang diunggah
    img = Image.open(uploaded_file).convert("RGB")
    classify_image(img)
