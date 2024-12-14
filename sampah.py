import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import os

# Definisikan kelas sampah
CLASSES = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash',  
           'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Fungsi untuk memuat model yang sudah dilatih
def load_model(model_path):
    # Memuat ResNet50
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))  # Output layer sesuai jumlah kelas

    # Periksa apakah file model tersedia
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Memuat model dengan mapping ke CPU
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model ke mode evaluasi
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


# Fungsi untuk memproses gambar yang di-upload
def process_image(image):
    # Transformasi yang diperlukan untuk gambar input
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ResNet normalization
    ])
    img_tensor = preprocess(image)
    st.write("Tensor Input Shape:", img_tensor.shape)  # Debugging
    img_tensor = img_tensor.unsqueeze(0)  # Tambah dimensi batch
    return img_tensor

# Fungsi untuk melakukan prediksi
def predict(image, model):
    img_tensor = process_image(image)
    with torch.no_grad():  # Menonaktifkan gradient tracking
        output = model(img_tensor)  # Melakukan prediksi
        st.write("Model Output:", output)  # Debugging
        
    _, predicted_class = torch.max(output, 1)
    st.write("Predicted Index:", predicted_class.item())  # Debugging
    return CLASSES[predicted_class.item()]

# Main Streamlit UI
def main():
    st.title("Prediksi Kelas Sampah Menggunakan ResNet50")
    
    # Upload gambar
    uploaded_file = st.file_uploader("Upload Gambar Sampah", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang di-upload", use_column_width=True)

        # Muat model
        model_path = "modelResNet50_model.pth"  # Sesuaikan path model Anda
        if os.path.exists(model_path):
            model = load_model(model_path)
            
            # Prediksi kelas
            predicted_class = predict(image, model)
            st.write(f"Prediksi Kelas Sampah: {predicted_class}")
        else:
            st.error("Model tidak ditemukan. Pastikan file modelresnet.pth ada di direktori yang sesuai.")

if __name__ == "__main__":
    main()
