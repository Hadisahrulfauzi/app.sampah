import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os
import io

# Sidebar untuk memilih halaman
menu = st.sidebar.radio("Pilih Halaman", ["Beranda", "Kamera", "Riwayat"])

# Memuat model yang sudah dilatih
model_path = 'modelResNet50_model.pth'
if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di {model_path}")
else:
    # Membuat model ResNet50 tanpa bobot pretrained
    model = models.resnet50(pretrained=False)

    # Menyesuaikan layer fully connected (fc) untuk jumlah kelas yang benar (9 kelas)
    num_ftrs = model.fc.in_features  # Mendapatkan jumlah fitur input untuk fc layer
    model.fc = torch.nn.Linear(num_ftrs, 9)  # Menyesuaikan layer fc dengan 9 kelas
    
    try:
        # Memuat state_dict ke dalam model, namun dengan pengecualian untuk layer 'fc'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()  # Set model ke mode evaluasi
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()

    # Memuat nama kelas dari model (disesuaikan dengan jumlah kelas model)
    classes = ['gfdh', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

    # Fungsi untuk memproses gambar input
    def preprocess_image(img):
        preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisasi sesuai dengan model ImageNet
        ])
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Menambah dimensi batch
        return img_tensor

    # Fungsi untuk memprediksi gambar
    def predict_image(img_tensor):
        with torch.no_grad():  # Menonaktifkan tracking gradient
            output = model(img_tensor)  # Melakukan prediksi
        _, class_idx = torch.max(output, 1)  # Menentukan kelas dengan probabilitas tertinggi
        return classes[class_idx.item()], torch.nn.functional.softmax(output, dim=1)[0][class_idx.item()]  # Kembalikan kelas dan probabilitas

    # Menyimpan riwayat ke session state jika belum ada
    if "history" not in st.session_state:
        st.session_state.history = []

    # Header dengan gambar dan deskripsi
    st.image("https://example.com/path_to_header_image.jpg", use_container_width=True)  # Ganti dengan URL gambar header yang sesuai
    st.title("Deteksi Penyakit Tanaman dengan AI")

    if menu == "Beranda":
        st.markdown("""
        Aplikasi ini menggunakan model *PyTorch* untuk mendeteksi penyakit pada tanaman kentang.
        Gunakan menu Kamera untuk mengambil gambar daun tanaman dan memprediksi penyakit yang ada.
        """, unsafe_allow_html=True)

    elif menu == "Kamera":
        # Menampilkan pilihan untuk mengambil gambar menggunakan kamera
        camera_input = st.camera_input("Ambil gambar untuk diprediksi")
        
        # Menambahkan fitur untuk mengunggah gambar dari perangkat
        uploaded_file = st.file_uploader("Unggah gambar dari perangkat", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Jika gambar diunggah
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang diunggah.", use_container_width=True)

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
                    st.experimental_rerun()  # Me-refresh halaman setelah penghapusan
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
