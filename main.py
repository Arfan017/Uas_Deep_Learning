import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from streamlit_option_menu import option_menu

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('model.hdf5')
	return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction

model = load_model()

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ['Klasifikasi', 'Dataset'],
        default_index = 0,
        
    )

if selected == 'Klasifikasi':
    st.title("Klasifikasi gambar kuyit, lengkuas, jahe")
    st.text("Upload gambar kuyit, lengkuas atau jahe untuk mulai klasifikasi gambar")

    uploaded_file = st.file_uploader("Pilih gambar kuyit, lengkuas atau jahe ...", type="jpg")
    if (uploaded_file is not None):
        image = Image.open(uploaded_file)
        st.image(image, caption='Upload gambar.', use_column_width=True)
        st.write("")
        
        with st.spinner("Klasifikasi....."):

            pred = predict_class(np.asarray(image), model)
            score = tf.nn.softmax(pred)
            time.sleep(1)
            Class_Names = ['jahe', 'kunyit', 'lengkuas']
            result = Class_Names[np.argmax(pred)]
            # result.data
            output = "Ini gambar {} dengan persentase: {:.2f}".format(result, 100 * np.max(pred))
            st.success(output)
            
if selected == 'Dataset':
    st.title("Dataset")
    st.text("Dataset yang digunakan merupaka hasil pengumpulan gambar sendiri")
    st.text("Dataset terdiri dari 3 Class yaitu: \nClass ke-1 : Kunyit \nClass ke-2 : Lengkuas \nClass ke-3 : Jahe")
    st.text("Masing-Masing class memiliki 100 gambar, total gambar 300")

    st.text("*Epoch*\njumlah Epoch yang digunakan 150")
    st.text("*Batch Size*\njumlah Epoch yang digunakan 64")