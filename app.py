import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from PIL import Image
import os

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_efficient_model():
    model = load_model('best_efficient_local.keras')
    return model

model = load_efficient_model()

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def generate_gradcam(model, img_array, class_idx):
    grad_model = Model(
        inputs=model.input,
        outputs=[
            model.get_layer('top_conv').output,
            model.output
        ]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def predict_tumor(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_array = np.expand_dims(img.copy(), axis=0)
    img_array = preprocess_input(img_array.astype('float32'))

    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    heatmap = generate_gradcam(model, img_array, predicted_idx)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    return predictions[0], predicted_idx, confidence, img, heatmap_resized, superimposed


st.title("🧠 NeuroScan AI")
st.subheader("Brain Tumor Detection & Classification System")

st.markdown("### 📤 Upload MRI Scan")
uploaded_file = st.file_uploader("Choose an MRI image",type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    with st.spinner('🔍 Analyzing MRI scan...'):
        predictions, predicted_idx, confidence, \
        img_array, heatmap, superimposed = predict_tumor(image)

    st.markdown(" ### 🔬 Analysis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original MRI**")
        st.image(img_array, use_container_width=True)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        fig, ax = plt.subplots()
        ax.imshow(heatmap, cmap='jet')
        ax.axis('off')
        st.pyplot(fig)

    with col3:
        st.markdown("**Overlay**")
        st.image(superimposed, use_container_width=True)

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("### 🎯 Prediction")

        if predicted_idx == 0:
            st.error(f"🔴 GLIOMA — {confidence:.2f}%")
        elif predicted_idx == 1:
            st.warning(f"🟡 MENINGIOMA — {confidence:.2f}%")
        elif predicted_idx == 2:
            st.success(f"🟢 NO TUMOR — {confidence:.2f}%")
        else:
            st.info(f"🔵 PITUITARY — {confidence:.2f}%")

    with col5:
        st.markdown("### 📊 Class Probabilities")
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            st.progress(float(prob), text=f"{cls}: {prob*100:.2f}%")

    st.markdown("---")
    st.markdown("⚠️ **Disclaimer:** This tool is for research purposes only. Always consult a qualified medical professional for diagnosis.")