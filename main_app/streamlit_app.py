import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


st.title(":green[Cataract Scan]")
st.image("/workspaces/cataract_classifier/main_app/cataract.jpg", width = 900)

st.sidebar.title("About")
st.sidebar.subheader("This app was built on a machine learning model that is able to detect both early and mature stages of cataract.")
st.sidebar.subheader("It is a knowledge showcase of our journey so far as cohort fellows of the Three Million Technical Talent (3MTT) program.")



# st.subheader("Overview")
# st.divider()
st.subheader("What is cataract?")
st.text("A cataract is an opacity in the clear lens. Normally, the human lens converges light rays. An opacity in the lens will scatter or block the light rays. If the opacity is small and at the lens periphery, there will be little or no interference with vision. On the other hand, when the opacity is central and dense, the light rays can be severely interfered with. This will lead to blurred vision.")
st.subheader("Causes of Cataract")
st.text("The most common cause of cataract is old age and this is known as senile cataract. Other causes are: trauma, drug toxicity (steroid), metabolic diseases (diabetes and hypoparathyrodism) and ocular diseases (uveitis and retinal detachment).")
st.text("Based on the stages of development, cataract can be mature( advanced stage with significant vision loss) and  immature( early stage with some signs of vision loss).")
st.subheader("Do you notice your vision blurry, don't panic. Upload your eye image after which you must book appointment with an Doctor eye if need be.")

# Load the model
@st.cache_resource
def load_cataract_model():
    return load_model("/workspaces/cataract_classifier/main_app/eye_model.keras")

model = load_cataract_model()
class_names = ["Immature", "Mature", "Normal"]  

## upload image
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption='Uploaded Image',  use_container_width =True)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write("Prediction:", predicted_class)