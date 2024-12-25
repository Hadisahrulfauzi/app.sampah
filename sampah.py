import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import io  # Untuk menangani gambar dalam memori

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Klasifikasi Sampah", "Riwayat Hasil Klasifikasi"])

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

    # Memuat nama kelas dari model (disesuaikan dengan jumlah kelas model) - Nama Kelas dalam Bahasa Indonesia
    classes = ['Kardus', 'Organik Makanan', 'Kaca', 'Logam', 'Sampah Campuran', 'Kertas', 'Plastik', 'Sampah Tekstil', 'Vegetasi']

    # Informasi tentang cara mendaur ulang sampah (dalam Bahasa Indonesia)
    recycling_info = {
    'Kardus': {
        'info': "Kardus dapat didaur ulang menjadi kertas daur ulang, kotak, dan produk lainnya.",
        'type': 'Non-Organik',
        'disposal': "Buang ke tempat sampah khusus kardus atau pusat daur ulang."
    },
    'Organik Makanan': {
        'info': "Organik makanan bisa diolah menjadi kompos atau digunakan untuk pembuatan energi.",
        'type': 'Organik',
        'disposal': "Buang ke tempat sampah organik atau fasilitas pengomposan."
    },
    'Kaca': {
        'info': "Kaca dapat didaur ulang menjadi produk kaca baru tanpa kehilangan kualitas.",
        'type': 'Non-Organik',
        'disposal': "Buang ke tempat sampah khusus kaca atau fasilitas daur ulang kaca."
    },
    'Logam': {
        'info': "Logam seperti aluminium dan besi dapat didaur ulang tanpa kehilangan kualitas dan digunakan kembali dalam berbagai produk.",
        'type': 'Non-Organik',
        'disposal': "Buang ke tempat sampah khusus logam atau pusat daur ulang logam."
    },
    'Sampah Campuran': {
        'info': "Sampah campuran sulit didaur ulang. Sebaiknya pisahkan komponen yang dapat didaur ulang.",
        'type': 'Non-Organik',
        'disposal': "Buang ke tempat sampah umum, namun usahakan untuk memisahkan terlebih dahulu bagian yang dapat didaur ulang."
    },
    'Kertas': {
        'info': "Kertas dapat didaur ulang menjadi produk kertas baru.",
        'type': 'Non-Organik',
        'disposal': "Buang ke tempat sampah khusus kertas atau pusat daur ulang."
    },
    'Plastik': {
        'info': "Plastik dapat didaur ulang menjadi berbagai produk baru, seperti bahan bangunan, tas, atau botol baru.",
        'type': 'Non-Organik',
        'disposal': "Buang ke tempat sampah khusus plastik atau fasilitas daur ulang plastik."
    },
    'Sampah Tekstil': {
        'info': "Pakaian dan kain bekas bisa didaur ulang menjadi bahan baru atau digunakan kembali dalam pembuatan produk tekstil lainnya.",
        'type': 'Non-Organik',
        'disposal': "Buang ke pusat pengumpulan pakaian bekas atau tempat daur ulang tekstil."
    },
    'Vegetasi': {
        'info': "Tanaman dan vegetasi dapat diolah menjadi kompos atau digunakan untuk energi terbarukan.",
        'type': 'Organik',
        'disposal': "Buang ke tempat sampah organik atau fasilitas pengomposan."
    }
}


    # Fungsi untuk memproses gambar input
    def preprocess_image(img):
        img = img.resize((224, 224))  # Mengubah ukuran gambar sesuai input model
        img_array = np.array(img) / 255.0  # Normalisasi gambar
        img_array = np.transpose(img_array, (2, 0, 1))  # Menukar dimensi ke format C x H x W
        img_array = torch.tensor(img_array, dtype=torch.float32)  # Convert numpy array to tensor
        img_array = img_array.unsqueeze(0)  # Menambah dimensi batch menjadi [1, 3, 224, 224]
        return img_array

    # Fungsi untuk memprediksi gambar
    def predict_image(img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)  # Prediksi menggunakan model
        class_idx = torch.argmax(preds, dim=1)  # Menentukan kelas dengan probabilitas tertinggi
        predicted_class = classes[class_idx.item()]  # Kelas yang diprediksi
        confidence = torch.softmax(preds, dim=1)[0][class_idx].item()  # Mendapatkan probabilitas
        recycling_tip = recycling_info.get(predicted_class, {"info": "Informasi daur ulang tidak tersedia.", "disposal": "Informasi tidak tersedia.", "type": "Unknown"})
        return predicted_class, confidence, recycling_tip['info'], recycling_tip['disposal'], recycling_tip['type']


    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header dengan gambar dan deskripsi
    st.title("Aplikasi Klasifikasi Sampah")

    if menu == "Beranda":
        st.markdown("""
        Sampah merupakan masalah lingkungan yang semakin kompleks akibat pertumbuhan populasi dan aktivitas industri. Pengelolaan sampah yang buruk dapat menyebabkan pencemaran dan membahayakan kesehatan. Solusi yang diusulkan adalah penggunaan teknologi Convolutional Neural Network (CNN) untuk mengklasifikasikan sampah secara otomatis melalui gambar. Sistem ini akan membantu masyarakat memisahkan sampah dengan lebih akurat dan memberikan panduan tentang cara mendaur ulang atau membuang sampah dengan benar. Dengan demikian, sistem ini diharapkan meningkatkan tingkat daur ulang, mengurangi beban TPA, dan mendukung ekonomi sirkular serta pengelolaan sampah berkelanjutan.
        """, unsafe_allow_html=True)
        st.header("Kelompok 4")
        st.markdown("Muhammad Ridwan Wibisono (211351098)", unsafe_allow_html=True)
        st.markdown("Hadi Sahrul Fauzi (211351060)", unsafe_allow_html=True)
        st.markdown("Mochammad Revan B (211351084)", unsafe_allow_html=True)
        st.markdown("Abriel Salsabina P.Y (211351001)", unsafe_allow_html=True)

    elif menu == "Klasifikasi Sampah":
        input_choice = st.radio("Pilih Metode Input", ["Ambil Gambar dari Kamera", "Unggah Gambar dari Perangkat"])

        if input_choice == "Ambil Gambar dari Kamera":
            camera_input = st.camera_input("Ambil gambar untuk diprediksi")
            if camera_input is not None:
                img = Image.open(camera_input)
                st.image(img, caption="Gambar yang diambil.", use_container_width=True)

                # Proses gambar dan prediksi
                img_tensor = preprocess_image(img)
                label, confidence, recycling_tip, recycling_type = predict_image(img_tensor)
                st.write(f"**Prediksi**:  {recycling_type}{label}")
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

        elif input_choice == "Unggah Gambar dari Perangkat":
            uploaded_image = st.file_uploader("Pilih gambar untuk diprediksi", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                img = Image.open(uploaded_image)
                st.image(img, caption="Gambar yang diunggah.", use_container_width=True)

                # Proses gambar dan prediksi
                img_tensor = preprocess_image(img)
                label, confidence, recycling_tip, recycling_disposial, recycling_type = predict_image(img_tensor)
                st.write(f"**Prediksi Jenis Sampah**:  {recycling_type} ({label})")
                st.write(f"**Probabilitas**: {confidence:.2f}")
                st.write(f"**Cara Daur Ulang**: {recycling_tip} {recycling_disposial}")
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

    elif menu == "Riwayat Hasil Klasifikasi":
        if len(st.session_state.history) == 0:
            st.write("Tidak ada riwayat prediksi.")
        else:
            st.write("Riwayat Prediksi Sampah:")

            for i, entry in enumerate(st.session_state.history):
                st.image(entry["image"], caption=f"Prediksi {i+1}: {entry['label']} (Probabilitas: {entry['confidence']:.2f})", use_container_width=True)
                st.write(f"*Prediksi*: {entry['label']}")
                st.write(f"*Probabilitas*: {entry['confidence']:.2f}")
                st.write(f"*Cara Daur Ulang*: {entry['recycling_tip']}")
                st.write(f"*Jenis Sampah*: {entry['recycling_type']}")

                if st.button(f"Hapus Prediksi {i+1}", key=f"hapus_{i}"):
                    st.session_state.history.pop(i)
                    st.rerun()  # Refresh halaman setelah penghapusan
                st.markdown("---")
