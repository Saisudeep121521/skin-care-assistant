import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Skin Care Assistant", page_icon="💆‍♀️", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATASET_PATH = "dataset"
CNN_MODEL_PATH = "skin_transfer_cnn.h5"
CLASS_LABELS = ["dry", "normal", "oily"]

# Skincare tips
SKINCARE_TIPS = {
    "dry": "💧 Your skin is dry. Use a gentle cleanser and apply moisturizer twice daily. Avoid hot showers.",
    "normal": "🌸 Your skin is normal. Maintain a simple routine: cleanser, moisturizer, and sunscreen daily.",
    "oily": "✨ Your skin is oily. Use a foaming cleanser, avoid heavy creams, and try oil-free moisturizers."
}

# -----------------------------
# LOAD DATA FOR COMPARISONS
# -----------------------------
@st.cache_resource
def load_data():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    return train_gen, val_gen

# -----------------------------
# FEATURE EXTRACTION FOR SVM/DT
# -----------------------------
def extract_features(generator):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128,128,3))
    feat_model = Model(inputs=base_model.input,
                       outputs=GlobalAveragePooling2D()(base_model.output))
    features, labels = [], []
    for x_batch, y_batch in generator:
        feat = feat_model.predict(x_batch, verbose=0)
        features.append(feat)
        labels.append(y_batch)
        if len(features) >= len(generator):
            break
    X = np.vstack(features)
    y = np.argmax(np.vstack(labels), axis=1)
    return X, y

# -----------------------------
# EVALUATE CNN
# -----------------------------
def evaluate_cnn(val_gen):
    cnn_model = load_model(CNN_MODEL_PATH)
    preds = cnn_model.predict(val_gen, verbose=0)
    Y_pred = np.argmax(preds, axis=1)
    Y_true = val_gen.classes
    acc = accuracy_score(Y_true, Y_pred)
    report = classification_report(Y_true, Y_pred, target_names=list(val_gen.class_indices.keys()), output_dict=True)
    return acc, report

# -----------------------------
# EVALUATE SVM & DT
# -----------------------------
def evaluate_sklearn_models(train_gen, val_gen):
    X_train, y_train = extract_features(train_gen)
    X_val, y_val = extract_features(val_gen)
    class_labels = list(train_gen.class_indices.keys())
    # SVM
    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_val)
    acc_svm = accuracy_score(y_val, y_pred_svm)
    report_svm = classification_report(y_val, y_pred_svm, target_names=class_labels, output_dict=True)
    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_val)
    acc_dt = accuracy_score(y_val, y_pred_dt)
    report_dt = classification_report(y_val, y_pred_dt, target_names=class_labels, output_dict=True)
    return (acc_svm, report_svm), (acc_dt, report_dt)

# -----------------------------
# LOAD MODEL FOR PREDICTIONS
# -----------------------------
@st.cache_resource
def load_skin_model():
    if not os.path.exists(CNN_MODEL_PATH):
        st.error("❌ CNN model not found! Please train the model first.")
        return None
    return load_model(CNN_MODEL_PATH)

model = load_skin_model()

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_skin_type(image):
    img = load_img(image, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]
    return CLASS_LABELS[class_index], confidence, predictions

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("💆‍♀️ AI-Powered Skin Care Assistant")
st.markdown("Upload your face image to detect **skin type** and explore our model comparisons.")

# Sidebar navigation
page = st.sidebar.radio("📑 Select Page", ["Skin Type Prediction", "Model Comparison"])

# ---- Skin Prediction Page ----
if page == "Skin Type Prediction":
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Analyze Skin"):
            with st.spinner("Analyzing..."):
                predicted_label, confidence, all_preds = predict_skin_type(uploaded_file)
            st.success(f"✨ Predicted Skin Type: **{predicted_label.upper()}**")
            st.info(f"Confidence: {confidence*100:.2f}%")
            st.subheader("💡 Skincare Recommendation")
            st.write(SKINCARE_TIPS[predicted_label])
            st.subheader("📊 Prediction Probabilities")
            st.bar_chart({CLASS_LABELS[i]: float(all_preds[i]) for i in range(len(CLASS_LABELS))})

# ---- Model Comparison Page ----
elif page == "Model Comparison":
    st.subheader("📊 Accuracy Comparison of Models")

    # Load data + evaluate
    with st.spinner("Evaluating models..."):
        train_gen, val_gen = load_data()
        cnn_acc, _ = evaluate_cnn(val_gen)
        (svm_acc, _), (dt_acc, _) = evaluate_sklearn_models(train_gen, val_gen)

    results = {
        "CNN": cnn_acc,
        "SVM": svm_acc,
        "Decision Tree": dt_acc
    }

    # Plot
    fig, ax = plt.subplots()
    models = list(results.keys())
    accuracies = list(results.values())
    bars = ax.bar(models, accuracies, color=['skyblue','lightgreen','salmon'])
    ax.set_ylim([0,1])
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", 
                ha="center", fontsize=10, fontweight="bold")
    st.pyplot(fig)

    best_model = max(results, key=results.get)
    st.success(f"✅ Best model: **{best_model}** with {results[best_model]*100:.2f}% accuracy.")
