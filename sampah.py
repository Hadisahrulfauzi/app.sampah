import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import io  # Untuk menangani gambar dalam memori

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Riwayat"])

# Memuat model yang sudah dilatih
model_path = 'modelResNet50_model.pth'
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

    # Informasi tentang cara mendaur ulang sampah
    recycling_info = {
        'Cardboard': {
            'info': "Kardus dapat didaur ulang menjadi kertas daur ulang, kotak, dan produk lainnya.",
            'type': 'Non-Organik'
        },
        'Food Organics': {
            'info': "Organik makanan bisa diolah menjadi kompos atau digunakan untuk pembuatan energi.",
            'type': 'Organik'
        },
        'Glass': {
            'info': "Kaca dapat didaur ulang menjadi produk kaca baru tanpa kehilangan kualitas.",
            'type': 'Non-Organik'
        },
        'Metal': {
            'info': "Logam seperti aluminium dan besi dapat didaur ulang tanpa kehilangan kualitas dan digunakan kembali dalam berbagai produk.",
            'type': 'Non-Organik'
        },
        'Miscellaneous Trash': {
            'info': "Sampah campuran sulit didaur ulang. Sebaiknya pisahkan komponen yang dapat didaur ulang.",
            'type': 'Non-Organik'
        },
        'Paper': {
            'info': "Kertas dapat didaur ulang menjadi produk kertas baru.",
            'type': 'Non-Organik'
        },
        'Plastic': {
            'info': "Plastik dapat didaur ulang menjadi berbagai produk baru, seperti bahan bangunan, tas, atau botol baru.",
            'type': 'Non-Organik'
        },
        'Textile Trash': {
            'info': "Pakaian dan kain bekas bisa didaur ulang menjadi bahan baru atau digunakan kembali dalam pembuatan produk tekstil lainnya.",
            'type': 'Non-Organik'
        },
        'Vegetation': {
            'info': "Tanaman dan vegetasi dapat diolah menjadi kompos atau digunakan untuk energi terbarukan.",
            'type': 'Organik'
        }
    }

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
            predicted_class = classes[class_idx.item()]
            confidence = torch.softmax(preds, dim=1)[0][class_idx].item()  # Mendapatkan probabilitas
            recycling_tip = recycling_info.get(predicted_class, {"info": "Informasi daur ulang tidak tersedia.", "type": "Unknown"})
            return predicted_class, confidence, recycling_tip['info'], recycling_tip['type']

    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header dengan gambar dan deskripsi
    st.image("https://example.com/path_to_header_image.jpg", use_container_width=True)  # Ganti dengan URL gambar header yang sesuai
    st.title("Klasifikasi Sampah")

    if menu == "Beranda":
        st.markdown("""
        Aplikasi ini menggunakan model *Pytorch* untuk mendeteksi jenis sampah dan memberikan informasi cara mendaur ulangnya.
        Gunakan menu Kamera untuk mengambil gambar sampah dan memprediksi jenisnya.
        """, unsafe_allow_html=True)

    elif menu == "Kamera":
        # Pilihan untuk mengambil gambar menggunakan kamera
        camera_input = st.camera_input("Ambil gambar untuk diprediksi")

        if camera_input is not None:
            # Menampilkan gambar yang diambil
            img = Image.open(camera_input)
            st.image(img, caption="Gambar yang diambil.", use_container_width=True)

            # Memproses gambar
            img_tensor = preprocess_image(img)

            # Prediksi
            label, confidence, recycling_tip, recycling_type = predict_image(img_tensor)
            st.write(f"**Prediksi**: {label}")
            st.write(f"**Probabilitas**: {confidence:.2f}")
            st.write(f"**Cara Daur Ulang**: {recycling_tip}")
            st.write(f"**Jenis Sampah**: {recycling_type}")

            # Menyimpan gambar dan hasil prediksi ke riwayat
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            st.session_state.history.append({
                "image": img_bytes,
                "label": label,
                "confidence": confidence,
                "recycling_tip": recycling_tip,
                "recycling_type": recycling_type
            })

    elif menu == "Riwayat":
        # Menampilkan riwayat hasil prediksi
        if len(st.session_state.history) == 0:
            st.write("Tidak ada riwayat prediksi.")
        else:
            st.write("Riwayat Prediksi Sampah:")

            # Loop untuk menampilkan setiap entri dalam riwayat
            for i, entry in enumerate(st.session_state.history):
                # Menampilkan gambar dari riwayat
                st.image(entry["image"], caption=f"Prediksi {i+1}: {entry['label']} (Probabilitas: {entry['confidence']:.2f})", use_container_width=True)
                st.write(f"*Prediksi*: {entry['label']}")
                st.write(f"*Probabilitas*: {entry['confidence']:.2f}")
                st.write(f"*Cara Daur Ulang*: {entry['recycling_tip']}")
                st.write(f"*Jenis Sampah*: {entry['recycling_type']}")

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
           
