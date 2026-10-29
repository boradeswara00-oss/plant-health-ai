import csv
import inspect
import io
from pathlib import Path
 
import streamlit as st
from PIL import Image
 
# ---------------- PAGE SETTINGS ----------------
# st.set_page_config must be the first Streamlit command and may be called
# only once, so it lives here at the top.
st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
from history_manager import (
    get_prediction_history,
    save_prediction,
    clear_prediction_history,
)
 
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "plant_disease_model_v2.keras"
CLASS_PATH = APP_DIR / "class_names_v2.json"
 
CONFIDENCE_THRESHOLD = 60  # scores below this (in %) are treated as uncertain
 
PLANT_OPTIONS = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Bell pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]
 
# Keyword that must appear in the model's plant name for each dropdown choice.
# Model names look like "Pepper, bell", "Corn (maize)" or "Cherry (including
# sour)", so we search for a keyword instead of comparing exact text.
PLANT_KEYWORDS = {
    "Apple": "apple",
    "Blueberry": "blueberry",
    "Cherry": "cherry",
    "Corn": "corn",
    "Grape": "grape",
    "Orange": "orange",
    "Peach": "peach",
    "Bell pepper": "pepper",
    "Potato": "potato",
    "Raspberry": "raspberry",
    "Soybean": "soybean",
    "Squash": "squash",
    "Strawberry": "strawberry",
    "Tomato": "tomato",
}
 
 
def stretch_button_args():
    """Full-width button arguments that work on old and new Streamlit."""
    try:
        if "width" in inspect.signature(st.button).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}
 
 
STRETCH = stretch_button_args()
 
 
def format_class_name(name):
    """Make model class names easier to read."""
    text = str(name).replace("___", " — ").replace("_", " ")
    return " ".join(text.split())
 
 
def plant_matches(selected, predicted_plant):
    """Check whether the predicted plant is the one the user selected."""
    keyword = PLANT_KEYWORDS.get(selected, selected.lower())
    return keyword in predicted_plant.lower()
 
 
def get_disease_guidance(class_name):
    """Return centralised guidance for supported plant conditions."""
 
    name = class_name.lower().replace("_", " ")
    plant = name.split("—")[0].strip() if "—" in name else ""
 
    general_care = [
        "Inspect both sides of the leaves for spots, discoloration, or pests.",
        "Monitor whether symptoms spread to other leaves.",
        "Avoid applying chemicals based only on an AI prediction.",
        "Consult a local agricultural expert if symptoms persist or worsen.",
    ]
 
    if "background" in name:
        return {
            "title": "No clear leaf detected",
            "about": (
                "The model detected a background class rather than a "
                "recognisable plant condition."
            ),
            "care": [
                "Photograph one whole leaf in focus.",
                "Use good natural lighting and a plain background.",
                "Make sure the leaf fills most of the image.",
            ],
        }
 
    if "healthy" in name:
        return {
            "title": f"Possible healthy {plant or 'plant'}",
            "about": (
                "The model classified the image as a healthy class. "
                "This does not guarantee the plant is free of disease."
            ),
            "care": [
                "Continue suitable watering for your plant.",
                "Provide appropriate light and well-drained soil.",
                "Inspect leaves regularly for changes or pests.",
            ],
        }
 
    # Central disease information for all supported plants.
    disease_rules = [
        (
            "apple scab",
            "Possible apple scab",
            "Can cause olive-coloured spots on leaves and fruit.",
        ),
        (
            "cedar apple rust",
            "Possible cedar apple rust",
            "May cause yellow-orange leaf spots and related symptoms.",
        ),
        (
            "black rot",
            "Possible black rot",
            "Can cause leaf lesions and other plant tissue damage.",
        ),
        (
            "powdery mildew",
            "Possible powdery mildew",
            "Often appears as white or grey powdery growth on leaves.",
        ),
        (
            "cercospora leaf spot",
            "Possible Cercospora leaf spot",
            "May cause leaf spots that expand and damage leaf tissue.",
        ),
        (
            "gray leaf spot",
            "Possible gray leaf spot",
            "Can produce elongated or rectangular lesions on leaves.",
        ),
        (
            "common rust",
            "Possible common rust",
            "Can produce reddish-brown or rust-coloured pustules on leaves.",
        ),
        (
            "northern leaf blight",
            "Possible northern leaf blight",
            "May produce long, grey-green or brown lesions on corn leaves.",
        ),
        (
            "esca",
            "Possible grape Esca disease",
            "Can cause leaf discoloration and other symptoms in grapevines.",
        ),
        (
            "leaf blight",
            "Possible leaf blight",
            "May cause leaf lesions, browning, and damaged leaf tissue.",
        ),
        (
            "haunglongbing",
            "Possible citrus greening",
            "Can cause uneven leaf mottling, yellowing, and poor fruit development.",
        ),
        (
            "citrus greening",
            "Possible citrus greening",
            "Can cause uneven leaf mottling, yellowing, and poor fruit development.",
        ),
        (
            "bacterial spot",
            "Possible bacterial spot",
            "May cause dark, water-soaked or necrotic spots on leaves.",
        ),
        (
            "early blight",
            f"Possible {plant or 'plant'} early blight",
            "May cause dark leaf spots, sometimes with concentric rings "
            "and surrounding yellowing.",
        ),
        (
            "late blight",
            f"Possible {plant or 'plant'} late blight",
            "Can cause rapidly expanding, water-soaked or dark lesions "
            "on leaves in favourable conditions.",
        ),
        (
            "leaf mold",
            "Possible leaf mold",
            "May cause pale yellow patches on upper leaf surfaces "
            "and mould growth underneath.",
        ),
        (
            "septoria leaf spot",
            "Possible Septoria leaf spot",
            "Often causes multiple small, dark-bordered spots, sometimes "
            "with pale centres.",
        ),
        (
            "spider mites",
            "Possible spider mite damage",
            "Can cause fine stippling, yellowing, and sometimes webbing.",
        ),
        (
            "target spot",
            "Possible target spot",
            "May produce brown lesions with concentric patterns.",
        ),
        (
            "yellow leaf curl virus",
            "Possible yellow leaf curl virus",
            "Can cause leaf curling, yellowing, stunted growth, and reduced yield.",
        ),
        (
            "mosaic virus",
            "Possible mosaic virus",
            "May cause mottled light and dark green areas or distorted leaves.",
        ),
        (
            "leaf scorch",
            "Possible leaf scorch",
            "Can cause browning or scorched-looking areas along leaf edges.",
        ),
    ]
 
    for keyword, title, about in disease_rules:
        if keyword in name:
            return {
                "title": title,
                "about": (
                    about + " The model prediction is not a confirmed diagnosis."
                ),
                "care": general_care,
            }
 
    return {
        "title": "Plant condition information",
        "about": (
            f"The model predicted: {class_name}. "
            "The plant or disease may have been misidentified. "
            "This result is not a confirmed diagnosis."
        ),
        "care": general_care,
    }
 
 
# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f3f9f3 0%, #ffffff 55%);
    }
 
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.8);
    }
 
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
 
    h2, h3 {
        color: #205c3b;
    }
 
    .hero {
        background: linear-gradient(120deg, #123d2b, #26734d);
        padding: 2.5rem 2rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 28px rgba(22, 78, 49, 0.15);
    }
 
    .hero h1 {
        color: white;
        font-size: 2.7rem;
        margin-bottom: 0.5rem;
    }
 
    .hero p {
        color: #e0f1e5;
        font-size: 1.08rem;
    }
 
    .section-title {
        color: #174b32;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1rem 0;
    }
 
    .info-card {
        background: white;
        border: 1px solid #dcebdd;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(30, 80, 45, 0.05);
    }
 
    .info-label {
        color: #64806b;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
 
    .info-value {
        color: #174b32;
        font-size: 1.2rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
 
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dce8d8;
        padding: 16px;
        border-radius: 12px;
    }
 
    div[data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #9bc5a2;
        border-radius: 16px;
        padding: 1rem;
    }
 
    .small-note {
        color: #617565;
        font-size: 0.9rem;
    }
 
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)
 
 
# ---------------- CHECK MODEL FILES ----------------
if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.error(
        "Model files are missing. Make sure "
        f"{MODEL_PATH.name} and {CLASS_PATH.name} "
        "are in the same folder as app.py."
    )
    st.stop()
 
 
# ---------------- LOAD PREDICTOR ----------------
try:
    from predictor import predict_plant_disease
except Exception as exc:
    st.error(f"Could not load the prediction module: {exc}")
    st.stop()
 
 
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🌿 PlantHealth AI")
    st.caption("Intelligent plant leaf analysis")
    st.divider()
 
    st.markdown("### How to use")
    st.write("1. Select the plant type.")
    st.write("2. Upload a clear leaf photo.")
    st.write("3. Click Analyze Leaf.")
    st.write("4. Review the predicted class.")
 
    st.divider()
    st.markdown("### Tips for better results")
    st.write("📸 Use a clear, focused photo.")
    st.write("🍃 Keep the leaf visible.")
    st.write("💡 Use good lighting.")
    st.write("🖼️ Avoid blurry images.")
 
    st.divider()
    st.caption("Educational AI prototype")
 
 
# ---------------- MAIN PAGE ----------------
st.markdown("""
<div class="hero">
    <h1>🌱 Healthy Plants Start Here</h1>
    <p>Use artificial intelligence to explore the likely health
    condition of a plant leaf.</p>
</div>
""", unsafe_allow_html=True)
 
st.info(
    "**Supported plants:** Apple, blueberry, cherry, corn, grape, orange, "
    "peach, bell pepper, potato, raspberry, soybean, squash, strawberry, "
    "and tomato.\n\n"
    "**Important:** Brinjal (eggplant) is not currently supported. "
    "The model may assign an unsupported plant to an incorrect class, "
    "even with a high confidence score. Predictions are estimates, "
    "not confirmed diagnoses."
)
 
left, right = st.columns([1, 1], gap="large")
 
 
with left:
    st.markdown(
        '<div class="section-title">📤 Upload a Leaf Image</div>',
        unsafe_allow_html=True,
    )
 
    selected_plant = st.selectbox(
        "🌿 Which plant are you uploading?",
        PLANT_OPTIONS,
        index=PLANT_OPTIONS.index("Tomato"),
        help="Select the plant shown in your image.",
    )
 
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Choose a clear image of a plant leaf.",
    )
 
    image = None
 
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded leaf")
        except Exception:
            st.error("This image could not be opened.")
            st.stop()
 
    analyze = st.button(
        "🔍 Analyze Leaf",
        type="primary",
        disabled=uploaded_file is None,
        **STRETCH,
    )
 
 
with right:
    st.markdown(
        '<div class="section-title">🧪 Analysis Results</div>',
        unsafe_allow_html=True,
    )
 
    if analyze and uploaded_file is not None and image is not None:
        try:
            with st.spinner("Analyzing your leaf image..."):
                best_class, confidence, top_predictions = (
                    predict_plant_disease(image)
                )
 
            # Save the prediction permanently in SQLite.
            save_prediction(
                uploaded_file.name,
                best_class,
                confidence,
            )
 
            display_class = format_class_name(best_class)
            predicted_plant = display_class.split("—")[0].strip()
 
            is_background = "background" in best_class.lower()
            is_healthy = "healthy" in best_class.lower()
            plant_mismatch = (
                not is_background
                and not plant_matches(selected_plant, predicted_plant)
            )
            low_confidence = confidence < CONFIDENCE_THRESHOLD
 
            if plant_mismatch:
                st.error(
                    f"PLANT MISMATCH DETECTED!\n\n"
                    f"You selected: {selected_plant}\n\n"
                    f"AI predicted: {predicted_plant}\n\n"
                    "The model may have identified the wrong plant. "
                    "Do not rely on this disease prediction. "
                    "Check the selected plant and upload a clearer image.",
                    icon="🚨",
                )
 
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">MOST LIKELY CLASS</div>
                    <div class="info-value">{display_class}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            st.write("")
 
            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">MODEL SCORE</div>
                    <div class="info-value">{confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
            st.write("")
            st.markdown("#### Top 3 predictions")
 
            for prediction in top_predictions:
                label = format_class_name(prediction["disease"])
                score = prediction["confidence"]
 
                st.write(f"**{label}**")
                st.progress(min(max(score / 100, 0.0), 1.0))
                st.caption(f"Model score: {score:.2f}%")
 
            if is_background:
                st.warning(
                    "This image may not show a clear plant leaf. "
                    "Try uploading a closer photo of one leaf with good lighting."
                )
            elif low_confidence:
                st.warning(
                    "The prediction is uncertain. Try a clearer, closer photo "
                    "of the leaf. Do not rely on this result alone to diagnose "
                    "a plant disease."
                )
            elif is_healthy and not plant_mismatch:
                st.success(
                    "The model predicts a healthy class, but this is not "
                    "a guarantee that the plant is disease-free."
                )
            elif not plant_mismatch:
                st.info(
                    "The model predicts a possible disease class. "
                    "Check the plant carefully and seek expert advice "
                    "if it appears unwell."
                )
 
            st.caption(
                "Model confidence is not the same as accuracy. Predictions can "
                "be wrong, especially for unclear images or plant types the "
                "model has not learned."
            )
 
            if plant_mismatch:
                guidance = {
                    "title": "Disease guidance withheld — plant mismatch",
                    "about": (
                        f"You selected {selected_plant}, but the model "
                        f"predicted {predicted_plant}. The disease prediction "
                        "may be incorrect, so plant-specific disease guidance "
                        "is not shown."
                    ),
                    "care": [
                        f"Confirm that the selected plant is {predicted_plant} "
                        "only if you can independently identify it.",
                        "Upload a clear photo showing one whole leaf.",
                        "Avoid applying disease-specific chemicals based only "
                        "on this prediction.",
                        "Consult a local agricultural expert if symptoms "
                        "persist or worsen.",
                    ],
                }
            elif low_confidence and not is_background:
                guidance = {
                    "title": "Unable to identify the condition reliably",
                    "about": (
                        f"The model's top prediction was {display_class}, "
                        f"with a score of {confidence:.2f}%. "
                        "The plant or disease may have been misidentified. "
                        "This is not a confirmed diagnosis."
                    ),
                    "care": [
                        "Take a clear, close-up photo of the affected leaf.",
                        "Include the whole leaf and its edges in the image.",
                        "Check both sides of the leaf for spots or "
                        "discoloration.",
                        "Avoid applying disease-specific chemicals based only "
                        "on this result.",
                        "Ask a local agricultural expert if symptoms persist "
                        "or spread.",
                    ],
                }
            else:
                guidance = get_disease_guidance(display_class)
 
            st.markdown("### Disease information and care tips")
            st.markdown(f"**{guidance['title']}**")
            st.write(guidance["about"])
 
            st.markdown("#### Recommended next steps")
            care_list = "\n".join(f"- {tip}" for tip in guidance["care"])
            st.markdown(care_list)
 
        except Exception as exc:
            st.error(f"Could not analyze this image: {exc}")
 
    else:
        st.info(
            "Your prediction will appear here after you upload an "
            "image and click **Analyze Leaf**."
        )
 
 
# ---------------- LOAD SAVED HISTORY (ONCE) ----------------
try:
    history = get_prediction_history()
except Exception as exc:
    history = []
    st.error(f"Could not load prediction history: {exc}")
 
 
# ---------------- ANALYTICS DASHBOARD ----------------
st.divider()
st.markdown("## 📊 Plant Health Analytics")
 
try:
    if history:
        total_predictions = len(history)
 
        average_confidence = (
            sum(record["Confidence"] for record in history)
            / total_predictions
        )
 
        disease_counts = {}
 
        for record in history:
            prediction_name = format_class_name(record["Prediction"])
            disease_counts[prediction_name] = (
                disease_counts.get(prediction_name, 0) + 1
            )
 
        most_common_prediction = max(
            disease_counts, key=disease_counts.get
        )
 
        col1, col2, col3 = st.columns(3)
 
        with col1:
            st.metric("Total Predictions", total_predictions)
 
        with col2:
            st.metric("Average Model Score", f"{average_confidence:.2f}%")
 
        with col3:
            st.metric(
                "Most Frequent Result",
                most_common_prediction,
                help="The class predicted most often in your saved history.",
            )
 
        st.markdown("### Top 5 Most Frequent Predictions")
 
        top_five = sorted(
            disease_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
 
        chart_data = {
            "Predicted Class": [item[0] for item in top_five],
            "Number of Predictions": [item[1] for item in top_five],
        }
 
        try:
            st.bar_chart(
                chart_data,
                x="Predicted Class",
                y="Number of Predictions",
                horizontal=True,
            )
        except TypeError:
            # Older Streamlit versions do not support horizontal charts.
            st.bar_chart(
                chart_data,
                x="Predicted Class",
                y="Number of Predictions",
            )
 
        st.caption(
            "These statistics describe your saved prediction history. "
            "They do not measure the model's real-world accuracy."
        )
 
    else:
        st.info(
            "Your analytics will appear after you analyze and save "
            "your first leaf prediction."
        )
 
except Exception as exc:
    st.error(f"Could not calculate analytics: {exc}")
 
 
# ---------------- PERMANENT PREDICTION HISTORY ----------------
st.divider()
st.markdown("## 📋 Prediction History")
st.caption(
    "Saved in a local SQLite database. History remains available "
    "when you restart the app."
)
 
try:
    if history:
        display_history = []
 
        for record in history:
            display_history.append({
                "Time": record["Time"],
                "Image": record["Image"],
                "Prediction": format_class_name(record["Prediction"]),
                "Confidence": f'{record["Confidence"]:.2f}%',
            })
 
        st.dataframe(display_history, hide_index=True)
 
        if st.button("🗑️ Clear History"):
            clear_prediction_history()
            st.success("Saved prediction history has been cleared.")
            st.rerun()
    else:
        st.info(
            "No saved predictions yet. Upload a leaf image and "
            "click Analyze Leaf to start your history."
        )
 
except Exception as exc:
    st.error(f"Could not display prediction history: {exc}")
 
 
# ---------------- DOWNLOAD HISTORY ----------------
if history:
    try:
        csv_buffer = io.StringIO()
 
        fieldnames = [
            "Prediction Time",
            "Image Name",
            "Predicted Disease",
            "Confidence (%)",
        ]
 
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
 
        for record in history:
            writer.writerow({
                "Prediction Time": record["Time"],
                "Image Name": record["Image"],
                "Predicted Disease": format_class_name(record["Prediction"]),
                "Confidence (%)": f"{record['Confidence']:.2f}",
            })
 
        st.download_button(
            label="📥 Download Prediction History (CSV)",
            data=csv_buffer.getvalue(),
            file_name="plant_health_prediction_history.csv",
            mime="text/csv",
            **STRETCH,
        )
    except Exception as exc:
        st.error(f"Could not prepare the CSV download: {exc}")
 
 
# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    """
    <p class="small-note" style="text-align:center;">
        🌿 PlantHealth AI · Educational project ·
        AI predictions can be incorrect
    </p>
    """,
    unsafe_allow_html=True,
)
 