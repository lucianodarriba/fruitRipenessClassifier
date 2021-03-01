import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import base64
from pathlib import Path
import matplotlib.pyplot as plt
# import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model(kerasModelFile):
    model = tf.keras.models.load_model(kerasModelFile, compile=False) #Directorio con el modelo
    return model

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def importClasses(labelFile):
    file = open(labelFile, "r")
    classes = []
    for line in file:
        classes.append(line[1:])
    
    file.close()
    return classes

def load_image(img):
    image = Image.open(img)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    return normalized_image_array

def plotBars(classes, prediction_ripeness):
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'ytick.color': 'white'})
    plt.rcParams.update({'ytick.labelsize': 40})
    plt.rcParams.update({'ytick.major.pad': 50})

    #arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    #ax.hist(arr, bins=20)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    pos = np.arange(len(classes))
    ax.set_xlim([0, 100])

    rects = ax.barh(pos, 100*prediction_ripeness[0], tick_label=classes, height=0.5, color='red', align='center')
    #ax.barh(pos, [100*prediction_ripeness[0][j] for j in range(len(classes))])
    
    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = float("{0:.1f}".format(rect.get_width()))
        rank_str = str(width) + " %"
        offset = 20
        # The bars aren't wide enough to print the ranking inside
        clr = 'white'
        if width < 40:
            # Shift the text to the right side of the right edge
            xloc = offset
            # Black against white background
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = -offset
            # White on magenta
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2
        label = ax.annotate(
            rank_str, xy=(width, yloc), xytext=(xloc, 0),
            textcoords="offset points",
            horizontalalignment=align, verticalalignment='center',
            color=clr, weight='bold', clip_on=True)
        rect_labels.append(label)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)
    st.pyplot(fig)


############################
# Main body of the web app #
############################

st.set_option('deprecation.showfileUploaderEncoding', False)
local_css("style.css")
header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width=150/>".format(img_to_bytes("baufest_logo.png"))
st.markdown(    header_html, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center'>Fruit ripeness classification</h1>", unsafe_allow_html=True)

with st.spinner('Cargando modelo en memoria ...'):
    model_fruitClassifier = load_model('keras_model_fruit_classifier.h5')
    model_banana = load_model('keras_model_banana.h5')
    model_avocado = load_model('keras_model_avocado.h5')
    model_peach = load_model('keras_model_peach.h5')
    classesFruitClassifier = importClasses("labels_fruit_classifier.txt")
    classesBanana = importClasses("labels_banana.txt")
    classesAvocado = importClasses("labels_avocado.txt")
    classesPeach = importClasses("labels_peach.txt")


uploaded_file = st.file_uploader("Subir imagen", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    
    ###content = uploaded_file.getvalue()
    decodedImage = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    decodedImage[0] = load_image(uploaded_file)
    # Load the image into the array

    with st.spinner('clasificando ...'):
        #First prediction step: Fruit type
        prediction_fruitType = model_fruitClassifier.predict(decodedImage)
        labelFruitType = np.argmax(prediction_fruitType,axis=1)
                
        #Second prediction step: Fruit ripeness
        if labelFruitType[0] == 0:
            prediction_ripeness = model_banana.predict(decodedImage)
            classes = classesBanana
        elif labelFruitType[0] == 1:
            prediction_ripeness = model_avocado.predict(decodedImage)
            classes = classesAvocado
        elif labelFruitType[0] == 2:
            prediction_ripeness = model_peach.predict(decodedImage)
            classes = classesPeach

        labelRipeness = np.argmax(prediction_ripeness,axis=1)

        st.markdown(f"**Fruta:** {classesFruitClassifier[labelFruitType[0]]}")
        st.markdown(f"**Maduración:** {classes[labelRipeness[0]]}")
        st.write("")

    #Display the fruit image and all the percentages from the classification        
    col1, mid, col2 = st.beta_columns([1,10,10])
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='', width=300, use_column_width=False)
    with col2:
        # dataChart = pd.DataFrame(100*prediction_ripeness[0])
        # st.bar_chart(data=dataChart)

        plotBars(classes, prediction_ripeness)

        # for j in range(len(classes)):
        #     st.write(f"{classes[j]}: {100*prediction_ripeness[0][j]:.1f}%")

