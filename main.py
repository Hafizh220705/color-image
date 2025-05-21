import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_colors(image, n_colors=5):
    image = image.convert('RGB')  # Pastikan RGB
    image = image.resize((200, 200))  # Resize for speed
    img_np = np.array(image)
    img_np = img_np.reshape((-1, 3))  # Ubah jadi array (N, 3)

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(img_np)

    colors = kmeans.cluster_centers_.astype(int)
    return colors

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# Judul Aplikasi
st.title("🎨 Dominant Color Picker from Image")

# Upload Gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Extracting dominant colors..."):
        colors = get_dominant_colors(image, n_colors=5)
        hex_colors = [rgb_to_hex(color) for color in colors]

    st.subheader("Dominant Color Palette")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.color_picker(f"Color {i+1}", hex_colors[i], label_visibility="collapsed")
            st.write(hex_colors[i])

# Informasi Identitas
st.markdown("---")
st.markdown("**Nama:** Hafizh Fadhl Muhammad  &nbsp;&nbsp;&nbsp;&nbsp; **NPM:** 140810230070")