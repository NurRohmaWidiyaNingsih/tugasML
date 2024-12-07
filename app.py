import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Memuat model yang sudah disimpan
model = joblib.load('cart_model.pkl')

# Desain Header dengan Logo Gambar
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            color: #4CAF50;
            font-weight: bold;
        }
        .subheader {
            text-align: center;
            font-size: 30px;
            color: #4CAF50;
        }
        .description {
            font-size: 18px;
            text-align: justify;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Menampilkan gambar/logo (pastikan gambar ada di direktori proyek)
st.image('logo.jpg', width=1000)  # Ganti dengan nama file logo Anda

# Menambahkan Navbar Dinamis dengan warna dan font yang menarik
page = st.sidebar.radio('Pilih Halaman', ('Prediksi Kanker Payudara', 'Visualisasi Tree'), key='page_nav')

# Halaman 1: Input Data dan Prediksi
if page == 'Prediksi Kanker Payudara':
    st.markdown('<h1 class="title">Prediksi Kanker Payudara</h1>', unsafe_allow_html=True)

    # Menambahkan Deskripsi
    st.markdown("""
        <div class="description">
            Masukkan nilai fitur dari sampel untuk memprediksi apakah kanker tersebut jinak atau ganas. 
            Cukup masukkan angka antara 1 hingga 10 untuk setiap fitur.
        </div>
    """, unsafe_allow_html=True)

    # Membuat layout dengan kolom input
    col1, col2, col3 = st.columns(3)

    with col1:
        clump_thickness = st.text_input('Clump Thickness', '1')
        uniformity_of_cell_size = st.text_input('Uniformity of Cell Size', '1')
        marginal_adhesion = st.text_input('Marginal Adhesion', '1')
        bare_nuclei = st.text_input('Bare Nuclei', '1')

    with col2:
        uniformity_of_cell_shape = st.text_input('Uniformity of Cell Shape', '1')
        single_epithelial_cell_size = st.text_input('Single Epithelial Cell Size', '1')
        bland_chromatin = st.text_input('Bland Chromatin', '1')
        normal_nucleoli = st.text_input('Normal Nucleoli', '1')

    with col3:
        mitoses = st.text_input('Mitoses', '1')

    # Tombol Prediksi
    if st.button('Prediksi', key="predict_btn"):
        try:
            # Convert input ke tipe data yang sesuai (angka)
            user_input = pd.DataFrame({
                'Clump_thickness': [int(clump_thickness)],
                'Uniformity_of_cell_size': [int(uniformity_of_cell_size)],
                'Uniformity_of_cell_shape': [int(uniformity_of_cell_shape)],
                'Marginal_adhesion': [int(marginal_adhesion)],
                'Single_epithelial_cell_size': [int(single_epithelial_cell_size)],
                'Bare_nuclei': [int(bare_nuclei)],
                'Bland_chromatin': [int(bland_chromatin)],
                'Normal_nucleoli': [int(normal_nucleoli)],
                'Mitoses': [int(mitoses)]
            })

            # Melakukan prediksi
            prediction = model.predict(user_input)
            prediction_label = 'Jinak (2)' if prediction[0] == 2 else 'Ganas (4)'

            # Menampilkan hasil prediksi
            st.markdown(f"**Hasil Prediksi: {prediction_label}**", unsafe_allow_html=True)

        except ValueError:
            st.error('Pastikan semua input adalah angka yang valid!')

# Halaman 2: Menampilkan Visualisasi Tree
elif page == 'Visualisasi Tree':
    st.markdown('<h1 class="title">Visualisasi Decision Tree</h1>', unsafe_allow_html=True)

    # Menampilkan visualisasi tree
    if st.button('Lihat Visualisasi Tree', key="visualize_tree"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model, filled=True, ax=ax)
        st.pyplot(fig)
