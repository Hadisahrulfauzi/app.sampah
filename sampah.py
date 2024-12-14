import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import io  # For handling image in memory

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Riwayat"])

# Memuat model yang sudah dilatih
model_path = 'D:\\PCD\\modelResNet50_model (1).pth'
if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di {model_path}")
else:
    # Inisialisasi model sesuai dengan arsitektur yang digunakan
    model = models.resnet50(pretrained=False)  # Tidak menggunakan model pretrained
    model.fc = nn.Linear(model.fc.in_features, 9)  # Sesuaikan dengan jumlah kelas yang ada (misal 9)

    # Memuat state_dict ke model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode

    # Memuat nama kelas dari model (disesuaikan dengan jumlah kelas model)
    classes = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']  # Ganti dengan nama kelas yang sesuai

    # Fungsi untuk memproses gambar input
    def preprocess_image(img):
        # Mengubah ukuran gambar sesuai input model
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalisasi gambar
        img_array = np.transpose(img_array, (2, 0, 1))  # Menukar dimensi ke format C x H x W
        img_array = torch.tensor(img_array, dtype=torch.float32)  # Convert numpy array to tensor
        img_array = img_array.unsqueeze(0)  # Menambah dimensi batch menjadi [1, 3, 224, 224]
        return img_array

    # Fungsi untuk memprediksi gambar
    def predict_image(img_tensor):
        with torch.no_grad():
            preds = model(img_tensor)  # Prediksi menggunakan model (seharusnya model langsung dipanggil)
            class_idx = torch.argmax(preds, dim=1)  # Menentukan kelas dengan probabilitas tertinggi
            return classes[class_idx.item()], torch.softmax(preds, dim=1)[0][class_idx].item()  # Mengembalikan kelas dan probabilitas

    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header dengan gambar dan deskripsi
    st.image("https://example.com/path_to_header_image.jpg", use_container_width=True)  # Ganti dengan URL gambar header yang sesuai
    st.title("Klasifikasi Sampah")

    if menu == "Beranda":
        st.markdown("""
        Aplikasi ini menggunakan model *Pytorch* untuk mendeteksi penyakit pada tanaman kentang.
        Gunakan menu Kamera untuk mengambil gambar daun tanaman dan memprediksi penyakit yang ada.
        """, unsafe_allow_html=True)

    elif menu == "Kamera":
        # Pilihan untuk mengambil gambar menggunakan kamera
        camera_input = st.camera_input("Ambil gambar untuk diprediksi")

        # Pilihan untuk mengupload gambar
        uploaded_image = st.file_uploader("Atau upload gambar", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Menampilkan gambar yang diupload
            img = Image.open(uploaded_image)
            st.image(img, caption="Gambar yang diupload.", use_container_width=True)

            # Memproses gambar
            img_tensor = preprocess_image(img)

            # Prediksi
            label, confidence = predict_image(img_tensor)
            st.write(f"Prediksi: {label}")
            st.write(f"Probabilitas: {confidence:.2f}")

            # Menyimpan gambar dan hasil prediksi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append({
                "image": img_bytes,
                "label": label,
                "confidence": confidence
            })

    elif menu == "Riwayat":
        # Menampilkan riwayat hasil prediksi
        if len(st.session_state.history) == 0:
            st.write("Tidak ada riwayat prediksi.")
        else:
            st.write("Riwayat Prediksi Penyakit Tanaman:")

            # Loop untuk menampilkan setiap entri dalam riwayat
            for i, entry in enumerate(st.session_state.history):
                # Menampilkan gambar dari riwayat
                st.image(entry["image"], caption=f"Prediksi {i+1}: {entry['label']} (Probabilitas: {entry['confidence']:.2f})", use_container_width=True)
                st.write(f"*Prediksi*: {entry['label']}")
                st.write(f"*Probabilitas*: {entry['confidence']:.2f}")

                # Menambahkan tombol hapus
                if st.button(f"Hapus Prediksi {i+1}", key=f"hapus_{i}"):
                    # Menghapus entri dari riwayat
                    st.session_state.history.pop(i)
                    st.rerun()  # Me-refresh halaman setelah penghapusan
                st.markdown("---")

# Menambahkan CSS kustom untuk mempercantik tampilan
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #6495ED;
            color: white;
            padding: 20px 0;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
        }
        .css-ffhzg2 {
            font-size: 1.25em;
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stImage>img {
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)
