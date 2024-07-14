import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score


json_file = open('mnist.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_new = model_from_json(loaded_model_json)

model_new.load_weights("mnist.h5")
print("Loaded model from disk")
# model_new =  keras.models.load_model("mnist.keras",compile=True)

st.title("MNIST Digit Recognizer")

SIZE = 192

canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150,width=150,
    drawing_mode='freedraw',
    key="canvas",
)

st.markdown(
    """
    <style>
        div[data-testid="stFileUploaderClear"] button {
            background-color: #ff0000;  /* Silme butonunun rengi */
        }
        div[data-testid="stFileUploaderUpload"] button {
            background-color: #00ff00;  /* İndirme butonunun rengi */
        }
    </style>
    """,
    unsafe_allow_html=True
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
    st.write(f'result: {np.argmax(pred[0])}')
    st.bar_chart(pred[0])
