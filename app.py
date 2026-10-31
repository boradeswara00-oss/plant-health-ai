import csv
import inspect
import io
import sqlite3
from datetime import date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageStat

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

# The PDF report is optional: the app still works if reportlab is missing.
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    HAS_PDF = True
except Exception:
    HAS_PDF = False

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "plant_disease_model_v2.keras"
CLASS_PATH = APP_DIR / "class_names_v2.json"
SAMPLES_DIR = APP_DIR / "samples"
GARDEN_DB = APP_DIR / "garden.db"

CONFIDENCE_THRESHOLD = 60  # scores below this (in %) are treated as uncertain
MAX_BATCH = 5
FOCUS_GRID = 4  # the leaf is split into 4 x 4 cells for the Focus Map
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC

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

PLANT_EMOJI = {
    "Apple": "🍎",
    "Blueberry": "🫐",
    "Cherry": "🍒",
    "Corn": "🌽",
    "Grape": "🍇",
    "Orange": "🍊",
    "Peach": "🍑",
    "Bell pepper": "🫑",
    "Potato": "🥔",
    "Raspberry": "🍓",
    "Soybean": "🌱",
    "Squash": "🎃",
    "Strawberry": "🍓",
    "Tomato": "🍅",
}

PLANT_HI = {
    "Apple": "सेब",
    "Blueberry": "ब्लूबेरी",
    "Cherry": "चेरी",
    "Corn": "मक्का",
    "Grape": "अंगूर",
    "Orange": "संतरा",
    "Peach": "आड़ू",
    "Bell pepper": "शिमला मिर्च",
    "Potato": "आलू",
    "Raspberry": "रास्पबेरी",
    "Soybean": "सोयाबीन",
    "Squash": "स्क्वैश",
    "Strawberry": "स्ट्रॉबेरी",
    "Tomato": "टमाटर",
}

# ---------------- TRANSLATIONS ----------------
STR = {
    "en": {
        "tab_analyze": "🔍 Analyze",
        "tab_stats": "📊 Analytics",
        "tab_history": "📋 History",
        "tab_library": "📚 Disease Library",
        "tab_batch": "🗂️ Batch",
        "tab_garden": "🪴 My Garden",
        "tab_pro": "🚀 Pro Tools",
        "tab_about": "ℹ️ About",
        "language": "🌐 Language / भाषा",
        "side_howto": "How to use",
        "side_1": "1. Select the plant type.",
        "side_2": "2. Upload a leaf photo, use the camera, or try a sample.",
        "side_3": "3. Click Analyze Leaf.",
        "side_4": "4. Review the result and download the report.",
        "side_tips": "Tips for better results",
        "side_tip1": "📸 Use a clear, focused photo.",
        "side_tip2": "🍃 Keep one whole leaf visible.",
        "side_tip3": "💡 Use good, even lighting.",
        "side_tip4": "🖼️ Avoid blurry images.",
        "side_caption": "Intelligent plant leaf analysis",
        "side_proto": "Educational AI prototype",
        "hero_h1": "🌱 Healthy Plants Start Here",
        "hero_p": "Use artificial intelligence to explore the likely health condition of a plant leaf.",
        "badge1": "🌿 14 plants",
        "badge2": "⚡ Instant results",
        "badge3": "📷 Camera or upload",
        "badge4": "📊 Saved history",
        "info_box": (
            "**Supported plants:** Apple, blueberry, cherry, corn, grape, orange, "
            "peach, bell pepper, potato, raspberry, soybean, squash, strawberry, "
            "and tomato.\n\n"
            "**Important:** Brinjal (eggplant) is not currently supported. "
            "The model may assign an unsupported plant to an incorrect class, "
            "even with a high confidence score. Predictions are estimates, "
            "not confirmed diagnoses."
        ),
        "sec_add": "📤 Add a Leaf Image",
        "which_plant": "🌿 Which plant are you uploading?",
        "which_plant_help": "Select the plant shown in your image.",
        "photo_source": "Photo source",
        "src_upload": "📁 Upload a file",
        "src_camera": "📷 Use camera",
        "src_sample": "🧪 Try a sample",
        "choose_file": "Choose a JPG or PNG image",
        "choose_help": "Choose a clear image of a plant leaf.",
        "take_photo": "Take a clear photo of one leaf",
        "pick_sample": "Pick a sample leaf",
        "your_photo": "Your leaf photo",
        "cant_open": "This image could not be opened.",
        "analyze_btn": "🔍 Analyze Leaf",
        "sec_results": "🧪 Analysis Results",
        "analyzing": "Analyzing your leaf image...",
        "analysis_err": "Could not analyze this image: {err}",
        "display_err": "Could not display the result: {err}",
        "prompt_result": "Your prediction will appear here after you add an image and click **Analyze Leaf**.",
        "clear_result": "🧹 Clear result",
        "most_likely": "MOST LIKELY CLASS",
        "model_score": "Model score",
        "selected_plant": "Selected plant",
        "top3": "Top 3 predictions",
        "score_caption": "Model score: {score:.2f}%",
        "st_mismatch": "Plant mismatch - do not rely on this result",
        "st_background": "No clear leaf detected",
        "st_uncertain": "Uncertain prediction",
        "st_healthy": "Predicted healthy",
        "st_disease": "Possible disease detected",
        "mismatch_msg": (
            "PLANT MISMATCH DETECTED!\n\nYou selected: {selected}\n\n"
            "AI predicted: {predicted}\n\n"
            "The model may have identified the wrong plant. "
            "Do not rely on this disease prediction. "
            "Check the selected plant and upload a clearer image."
        ),
        "bg_warn": "This image may not show a clear plant leaf. Try uploading a closer photo of one leaf with good lighting.",
        "low_warn": "The prediction is uncertain. Try a clearer, closer photo of the leaf. Do not rely on this result alone to diagnose a plant disease.",
        "healthy_ok": "The model predicts a healthy class, but this is not a guarantee that the plant is disease-free.",
        "disease_info": "The model predicts a possible disease class. Check the plant carefully and seek expert advice if it appears unwell.",
        "conf_note": "Model confidence is not the same as accuracy. Predictions can be wrong, especially for unclear images or plant types the model has not learned.",
        "care_header": "Disease information and care tips",
        "next_steps": "Recommended next steps",
        "dl_txt": "📄 Download this result (TXT)",
        "dl_pdf": "📑 Download this result (PDF)",
        "pdf_note": "The PDF report is written in English.",
        "garden_title": "🪴 Save to My Garden",
        "garden_name": "Name this plant (for example: Balcony tomato)",
        "garden_btn": "Save to My Garden",
        "garden_saved": "Saved to My Garden as '{name}'.",
        "garden_fail": "Could not save to My Garden: {err}",
        "q_small": "The image is small ({w}×{h} px). A larger photo usually gives a better result.",
        "q_dark": "The photo looks quite dark. Try better lighting.",
        "q_bright": "The photo looks very bright or washed out. Try less direct light.",
        "g_note": " The model prediction is not a confirmed diagnosis.",
        "g_general": [
            "Inspect both sides of the leaves for spots, discoloration, or pests.",
            "Monitor whether symptoms spread to other leaves.",
            "Avoid applying chemicals based only on an AI prediction.",
            "Consult a local agricultural expert if symptoms persist or worsen.",
        ],
        "g_bg_title": "No clear leaf detected",
        "g_bg_about": "The model detected a background class rather than a recognisable plant condition.",
        "g_bg_care": [
            "Photograph one whole leaf in focus.",
            "Use good natural lighting and a plain background.",
            "Make sure the leaf fills most of the image.",
        ],
        "g_healthy_title": "Possible healthy {plant}",
        "g_healthy_about": "The model classified the image as a healthy class. This does not guarantee the plant is free of disease.",
        "g_healthy_care": [
            "Continue suitable watering for your plant.",
            "Provide appropriate light and well-drained soil.",
            "Inspect leaves regularly for changes or pests.",
        ],
        "g_fallback_title": "Plant condition information",
        "g_fallback_about": "The model predicted: {cls}. The plant or disease may have been misidentified. This result is not a confirmed diagnosis.",
        "g_mm_title": "Disease guidance withheld — plant mismatch",
        "g_mm_about": "You selected {selected}, but the model predicted {predicted}. The disease prediction may be incorrect, so plant-specific disease guidance is not shown.",
        "g_mm_care": [
            "Confirm that the selected plant is {predicted} only if you can independently identify it.",
            "Upload a clear photo showing one whole leaf.",
            "Avoid applying disease-specific chemicals based only on this prediction.",
            "Consult a local agricultural expert if symptoms persist or worsen.",
        ],
        "g_low_title": "Unable to identify the condition reliably",
        "g_low_about": "The model's top prediction was {cls}, with a score of {score:.2f}%. The plant or disease may have been misidentified. This is not a confirmed diagnosis.",
        "g_low_care": [
            "Take a clear, close-up photo of the affected leaf.",
            "Include the whole leaf and its edges in the image.",
            "Check both sides of the leaf for spots or discoloration.",
            "Avoid applying disease-specific chemicals based only on this result.",
            "Ask a local agricultural expert if symptoms persist or spread.",
        ],
        "stats_title": "📊 Plant Health Analytics",
        "m_total": "Total Predictions",
        "m_avg": "Average Model Score",
        "m_healthy": "Healthy Results",
        "m_common": "Most Frequent Result",
        "m_common_help": "The class predicted most often in your saved history.",
        "top5": "Top 5 Most Frequent Predictions",
        "col_class": "Predicted Class",
        "col_count": "Number of Predictions",
        "stats_note": "These statistics describe your saved prediction history. They do not measure the model's real-world accuracy.",
        "stats_empty": "Your analytics will appear after you analyze and save your first leaf prediction.",
        "stats_err": "Could not calculate analytics: {err}",
        "hist_title": "📋 Prediction History",
        "hist_caption": "Saved in a local SQLite database. History remains available when you restart the app.",
        "hist_search": "🔎 Search history",
        "hist_search_ph": "Type a plant, disease or image name...",
        "hist_showing": "Showing {n} of {total} saved predictions.",
        "hist_nomatch": "No saved predictions match your search.",
        "hist_confirm": "I want to delete all saved history",
        "hist_clear": "🗑️ Clear History",
        "hist_cleared": "Saved prediction history has been cleared.",
        "hist_empty": "No saved predictions yet. Add a leaf image and click Analyze Leaf to start your history.",
        "hist_err": "Could not display prediction history: {err}",
        "hist_load_err": "Could not load prediction history: {err}",
        "hist_dl": "📥 Download Prediction History (CSV)",
        "csv_err": "Could not prepare the CSV download: {err}",
        "col_time": "Time",
        "col_image": "Image",
        "col_pred": "Prediction",
        "col_conf": "Confidence",
        "lib_title": "📚 Plant Disease Library",
        "lib_intro": "Learn about the conditions the model can recognise. This guide is for education only and is not a diagnosis.",
        "lib_search": "🔎 Search diseases",
        "lib_search_ph": "Type a disease or plant name...",
        "lib_filter": "Filter by plant",
        "lib_all": "All plants",
        "lib_affects": "Commonly seen on",
        "lib_symptoms": "What it may look like",
        "lib_none": "No conditions match your search.",
        "lib_prevent_title": "General prevention tips",
        "lib_prevent": [
            "Water at the base of the plant and avoid wetting the leaves.",
            "Leave enough space between plants for good air flow.",
            "Remove and safely dispose of badly affected leaves.",
            "Keep tools and hands clean when moving between plants.",
            "Rotate crops each season and choose resistant varieties where available.",
            "Ask a local agricultural expert before using any treatment.",
        ],
        "batch_title": "🗂️ Batch Analysis",
        "batch_intro": "Analyze up to {n} leaf photos at once. Every result is saved to your history.",
        "batch_plant": "🌿 Plant shown in these photos",
        "batch_upload": "Choose up to {n} images",
        "batch_limit": "Only the first {n} images will be analyzed.",
        "batch_btn": "🔍 Analyze all images",
        "batch_progress": "Analyzing image {i} of {n}...",
        "batch_fail": "{name}: could not be analyzed ({err})",
        "batch_summary": "Batch summary",
        "batch_dl": "📥 Download batch results (CSV)",
        "col_status": "Status",
        "col_score": "Score",
        "garden_page_title": "🪴 My Garden",
        "garden_intro": "Give each plant a name and save a check-in after every analysis. Watch its health over time.",
        "garden_empty": "Your garden is empty. Analyze a leaf, then use 'Save to My Garden' under the result.",
        "garden_pick": "Choose a plant",
        "garden_checkins": "Check-ins",
        "garden_last": "Last status",
        "garden_chart": "Health over time (1 = predicted healthy, 0 = needs attention)",
        "garden_col_date": "Check-in",
        "garden_col_health": "Health",
        "garden_col_status": "Result",
        "garden_ok": "Predicted healthy",
        "garden_bad": "Needs attention",
        "garden_confirm": "I want to delete this plant's check-ins",
        "garden_delete": "🗑️ Delete this plant",
        "garden_deleted": "Deleted '{name}' from My Garden.",
        "garden_note": "My Garden is stored next to the app. On free cloud hosting it can reset when the app restarts.",
        "about_title": "ℹ️ About PlantHealth AI",
        "about_intro": "PlantHealth AI uses a trained deep learning model to estimate the likely condition of a plant leaf from a photo. Every analysis is saved in a local SQLite database so you can review it later.",
        "about_plants": "Supported plants",
        "about_how": "How it works",
        "about_steps": (
            "1. You add a leaf photo.\n"
            "2. The image is prepared and passed to the model.\n"
            "3. The model scores every known class.\n"
            "4. The best matches are shown with care tips.\n"
            "5. The result is saved to your history."
        ),
        "about_good": "Good to know",
        "about_notes": (
            "- Results are estimates, not confirmed diagnoses.\n"
            "- Plants outside the supported list may be misclassified.\n"
            "- A high score does not guarantee a correct answer.\n"
            "- Disease names are shown in English.\n"
            "- Ask a local agricultural expert before applying any treatment."
        ),
        "footer": "🌿 PlantHealth AI · Educational project · AI predictions can be incorrect",
        "report_title": "PLANTHEALTH AI - LEAF ANALYSIS REPORT",
    },
    "hi": {
        "tab_analyze": "🔍 जाँच",
        "tab_stats": "📊 आँकड़े",
        "tab_history": "📋 इतिहास",
        "tab_library": "📚 रोग पुस्तकालय",
        "tab_batch": "🗂️ एक साथ जाँच",
        "tab_garden": "🪴 मेरा बगीचा",
        "tab_pro": "🚀 प्रो टूल्स",
        "tab_about": "ℹ️ जानकारी",
        "language": "🌐 Language / भाषा",
        "side_howto": "उपयोग कैसे करें",
        "side_1": "1. पौधे का प्रकार चुनें।",
        "side_2": "2. पत्ती की फ़ोटो अपलोड करें, कैमरा इस्तेमाल करें या नमूना चुनें।",
        "side_3": "3. 'पत्ती की जाँच करें' दबाएँ।",
        "side_4": "4. नतीजा देखें और रिपोर्ट डाउनलोड करें।",
        "side_tips": "बेहतर नतीजे के लिए सुझाव",
        "side_tip1": "📸 साफ़ और फ़ोकस वाली फ़ोटो लें।",
        "side_tip2": "🍃 एक पूरी पत्ती दिखाई दे।",
        "side_tip3": "💡 अच्छी और समान रोशनी रखें।",
        "side_tip4": "🖼️ धुंधली फ़ोटो से बचें।",
        "side_caption": "पौधों की पत्तियों का स्मार्ट विश्लेषण",
        "side_proto": "शैक्षिक एआई प्रोटोटाइप",
        "hero_h1": "🌱 स्वस्थ पौधों की शुरुआत यहाँ से",
        "hero_p": "कृत्रिम बुद्धिमत्ता की मदद से पौधे की पत्ती की संभावित सेहत जानें।",
        "badge1": "🌿 14 पौधे",
        "badge2": "⚡ तुरंत नतीजे",
        "badge3": "📷 कैमरा या अपलोड",
        "badge4": "📊 सहेजा हुआ इतिहास",
        "info_box": (
            "**समर्थित पौधे:** सेब, ब्लूबेरी, चेरी, मक्का, अंगूर, संतरा, "
            "आड़ू, शिमला मिर्च, आलू, रास्पबेरी, सोयाबीन, स्क्वैश, स्ट्रॉबेरी "
            "और टमाटर।\n\n"
            "**ज़रूरी:** बैंगन अभी समर्थित नहीं है। मॉडल किसी असमर्थित पौधे को "
            "ऊँचे स्कोर के साथ भी गलत श्रेणी में रख सकता है। नतीजे अनुमान हैं, "
            "पक्की जाँच नहीं।"
        ),
        "sec_add": "📤 पत्ती की फ़ोटो जोड़ें",
        "which_plant": "🌿 आप किस पौधे की फ़ोटो डाल रहे हैं?",
        "which_plant_help": "फ़ोटो में दिख रहा पौधा चुनें।",
        "photo_source": "फ़ोटो का स्रोत",
        "src_upload": "📁 फ़ाइल अपलोड करें",
        "src_camera": "📷 कैमरा इस्तेमाल करें",
        "src_sample": "🧪 नमूना आज़माएँ",
        "choose_file": "JPG या PNG फ़ोटो चुनें",
        "choose_help": "पत्ती की साफ़ फ़ोटो चुनें।",
        "take_photo": "एक पत्ती की साफ़ फ़ोटो लें",
        "pick_sample": "नमूना पत्ती चुनें",
        "your_photo": "आपकी पत्ती की फ़ोटो",
        "cant_open": "यह फ़ोटो खोली नहीं जा सकी।",
        "analyze_btn": "🔍 पत्ती की जाँच करें",
        "sec_results": "🧪 जाँच के नतीजे",
        "analyzing": "आपकी पत्ती की फ़ोटो की जाँच हो रही है...",
        "analysis_err": "इस फ़ोटो की जाँच नहीं हो सकी: {err}",
        "display_err": "नतीजा दिखाया नहीं जा सका: {err}",
        "prompt_result": "फ़ोटो जोड़कर **पत्ती की जाँच करें** दबाने के बाद आपका नतीजा यहाँ दिखेगा।",
        "clear_result": "🧹 नतीजा हटाएँ",
        "most_likely": "सबसे संभावित श्रेणी",
        "model_score": "मॉडल स्कोर",
        "selected_plant": "चुना गया पौधा",
        "top3": "शीर्ष 3 अनुमान",
        "score_caption": "मॉडल स्कोर: {score:.2f}%",
        "st_mismatch": "पौधा मेल नहीं खाता - इस नतीजे पर भरोसा न करें",
        "st_background": "साफ़ पत्ती नहीं मिली",
        "st_uncertain": "अनिश्चित अनुमान",
        "st_healthy": "स्वस्थ होने का अनुमान",
        "st_disease": "संभावित रोग मिला",
        "mismatch_msg": (
            "पौधा मेल नहीं खा रहा!\n\nआपने चुना: {selected}\n\n"
            "एआई का अनुमान: {predicted}\n\n"
            "मॉडल ने शायद गलत पौधा पहचाना है। इस रोग के अनुमान पर भरोसा न करें। "
            "चुना हुआ पौधा जाँचें और ज़्यादा साफ़ फ़ोटो अपलोड करें।"
        ),
        "bg_warn": "इस फ़ोटो में साफ़ पत्ती नहीं दिख रही। अच्छी रोशनी में एक पत्ती की नज़दीक से फ़ोटो लें।",
        "low_warn": "अनुमान अनिश्चित है। पत्ती की ज़्यादा साफ़ और नज़दीक की फ़ोटो लें। पौधे के रोग की पहचान के लिए सिर्फ़ इस नतीजे पर भरोसा न करें।",
        "healthy_ok": "मॉडल ने स्वस्थ श्रेणी बताई है, लेकिन इसका मतलब यह पक्का नहीं कि पौधा रोग-मुक्त है।",
        "disease_info": "मॉडल ने संभावित रोग श्रेणी बताई है। पौधे को ध्यान से देखें और बीमार लगे तो विशेषज्ञ की सलाह लें।",
        "conf_note": "मॉडल का भरोसा और सटीकता एक चीज़ नहीं हैं। अनुमान गलत हो सकते हैं, खासकर धुंधली फ़ोटो या ऐसे पौधों के लिए जिन्हें मॉडल ने नहीं सीखा।",
        "care_header": "रोग की जानकारी और देखभाल के सुझाव",
        "next_steps": "आगे क्या करें",
        "dl_txt": "📄 यह नतीजा डाउनलोड करें (TXT)",
        "dl_pdf": "📑 यह नतीजा डाउनलोड करें (PDF)",
        "pdf_note": "PDF रिपोर्ट अंग्रेज़ी में बनती है।",
        "garden_title": "🪴 मेरे बगीचे में सहेजें",
        "garden_name": "इस पौधे का नाम (जैसे: बालकनी का टमाटर)",
        "garden_btn": "मेरे बगीचे में सहेजें",
        "garden_saved": "'{name}' के नाम से मेरे बगीचे में सहेजा गया।",
        "garden_fail": "मेरे बगीचे में सहेजा नहीं जा सका: {err}",
        "q_small": "फ़ोटो छोटी है ({w}×{h} px)। बड़ी फ़ोटो से आमतौर पर बेहतर नतीजा मिलता है।",
        "q_dark": "फ़ोटो काफ़ी अँधेरी लग रही है। बेहतर रोशनी में लें।",
        "q_bright": "फ़ोटो बहुत चमकीली या फीकी लग रही है। सीधी तेज़ रोशनी से बचें।",
        "g_note": " मॉडल का अनुमान पक्की जाँच नहीं है।",
        "g_general": [
            "पत्तियों के दोनों तरफ़ धब्बे, रंग बदलना या कीड़े देखें।",
            "देखते रहें कि लक्षण दूसरी पत्तियों तक फैल रहे हैं या नहीं।",
            "सिर्फ़ एआई के अनुमान के आधार पर रसायन न डालें।",
            "लक्षण बने रहें या बढ़ें तो स्थानीय कृषि विशेषज्ञ से सलाह लें।",
        ],
        "g_bg_title": "साफ़ पत्ती नहीं मिली",
        "g_bg_about": "मॉडल को पौधे की पहचानी जा सकने वाली स्थिति की जगह पृष्ठभूमि (बैकग्राउंड) श्रेणी मिली।",
        "g_bg_care": [
            "एक पूरी पत्ती की फ़ोकस में फ़ोटो लें।",
            "अच्छी प्राकृतिक रोशनी और सादी पृष्ठभूमि रखें।",
            "पत्ती फ़ोटो के ज़्यादातर हिस्से में दिखे।",
        ],
        "g_healthy_title": "संभवतः स्वस्थ {plant}",
        "g_healthy_about": "मॉडल ने फ़ोटो को स्वस्थ श्रेणी में रखा है। इसका मतलब यह पक्का नहीं कि पौधा रोग-मुक्त है।",
        "g_healthy_care": [
            "पौधे को ज़रूरत के हिसाब से पानी देते रहें।",
            "सही रोशनी और अच्छी जल-निकासी वाली मिट्टी दें।",
            "बदलाव या कीड़ों के लिए पत्तियों को नियमित देखें।",
        ],
        "g_fallback_title": "पौधे की स्थिति की जानकारी",
        "g_fallback_about": "मॉडल का अनुमान: {cls}। हो सकता है पौधा या रोग गलत पहचाना गया हो। यह पक्की जाँच नहीं है।",
        "g_mm_title": "रोग की सलाह रोकी गई - पौधा मेल नहीं खाता",
        "g_mm_about": "आपने {selected} चुना, लेकिन मॉडल ने {predicted} बताया। रोग का अनुमान गलत हो सकता है, इसलिए पौधे-विशेष रोग की सलाह नहीं दिखाई गई।",
        "g_mm_care": [
            "चुना हुआ पौधा {predicted} तभी मानें जब आप खुद उसकी पहचान कर सकें।",
            "एक पूरी पत्ती की साफ़ फ़ोटो अपलोड करें।",
            "सिर्फ़ इस अनुमान पर रोग-विशेष रसायन न डालें।",
            "लक्षण बने रहें या बढ़ें तो स्थानीय कृषि विशेषज्ञ से सलाह लें।",
        ],
        "g_low_title": "स्थिति की भरोसेमंद पहचान नहीं हो सकी",
        "g_low_about": "मॉडल का सबसे ऊपर का अनुमान {cls} था, स्कोर {score:.2f}%। हो सकता है पौधा या रोग गलत पहचाना गया हो। यह पक्की जाँच नहीं है।",
        "g_low_care": [
            "प्रभावित पत्ती की साफ़, नज़दीक की फ़ोटो लें।",
            "फ़ोटो में पूरी पत्ती और उसके किनारे शामिल करें।",
            "पत्ती के दोनों तरफ़ धब्बे या रंग बदलना देखें।",
            "सिर्फ़ इस नतीजे पर रोग-विशेष रसायन न डालें।",
            "लक्षण बने रहें या फैलें तो स्थानीय कृषि विशेषज्ञ से पूछें।",
        ],
        "stats_title": "📊 पौधों की सेहत के आँकड़े",
        "m_total": "कुल अनुमान",
        "m_avg": "औसत मॉडल स्कोर",
        "m_healthy": "स्वस्थ नतीजे",
        "m_common": "सबसे ज़्यादा आया नतीजा",
        "m_common_help": "आपके सहेजे इतिहास में सबसे ज़्यादा बार बताई गई श्रेणी।",
        "top5": "सबसे ज़्यादा बार आए 5 अनुमान",
        "col_class": "अनुमानित श्रेणी",
        "col_count": "अनुमानों की संख्या",
        "stats_note": "ये आँकड़े आपके सहेजे इतिहास के हैं। ये मॉडल की असली सटीकता नहीं बताते।",
        "stats_empty": "पहली पत्ती की जाँच करने और सहेजने के बाद आँकड़े यहाँ दिखेंगे।",
        "stats_err": "आँकड़े निकाले नहीं जा सके: {err}",
        "hist_title": "📋 अनुमानों का इतिहास",
        "hist_caption": "स्थानीय SQLite डेटाबेस में सहेजा गया। ऐप दोबारा चालू करने पर भी इतिहास रहता है।",
        "hist_search": "🔎 इतिहास में खोजें",
        "hist_search_ph": "पौधे, रोग या फ़ोटो का नाम लिखें...",
        "hist_showing": "{total} में से {n} सहेजे अनुमान दिख रहे हैं।",
        "hist_nomatch": "आपकी खोज से कोई सहेजा अनुमान नहीं मिला।",
        "hist_confirm": "मैं सारा सहेजा इतिहास मिटाना चाहता/चाहती हूँ",
        "hist_clear": "🗑️ इतिहास मिटाएँ",
        "hist_cleared": "सहेजा हुआ इतिहास मिटा दिया गया।",
        "hist_empty": "अभी कोई अनुमान सहेजा नहीं गया। पत्ती की फ़ोटो जोड़कर जाँच करें, इतिहास शुरू हो जाएगा।",
        "hist_err": "इतिहास दिखाया नहीं जा सका: {err}",
        "hist_load_err": "इतिहास लोड नहीं हो सका: {err}",
        "hist_dl": "📥 इतिहास डाउनलोड करें (CSV)",
        "csv_err": "CSV डाउनलोड तैयार नहीं हो सका: {err}",
        "col_time": "समय",
        "col_image": "फ़ोटो",
        "col_pred": "अनुमान",
        "col_conf": "भरोसा",
        "lib_title": "📚 पौधों के रोगों का पुस्तकालय",
        "lib_intro": "जानें कि मॉडल कौन-सी स्थितियाँ पहचान सकता है। यह गाइड सिर्फ़ जानकारी के लिए है, पक्की जाँच नहीं।",
        "lib_search": "🔎 रोग खोजें",
        "lib_search_ph": "रोग या पौधे का नाम लिखें...",
        "lib_filter": "पौधे के अनुसार छाँटें",
        "lib_all": "सभी पौधे",
        "lib_affects": "आमतौर पर इन पर दिखता है",
        "lib_symptoms": "यह कैसा दिख सकता है",
        "lib_none": "आपकी खोज से कोई स्थिति नहीं मिली।",
        "lib_prevent_title": "बचाव के सामान्य सुझाव",
        "lib_prevent": [
            "पौधे की जड़ के पास पानी दें और पत्तियों को गीला करने से बचें।",
            "हवा के अच्छे बहाव के लिए पौधों के बीच पर्याप्त जगह रखें।",
            "बहुत प्रभावित पत्तियों को हटाकर सुरक्षित तरीके से नष्ट करें।",
            "एक पौधे से दूसरे पर जाते समय औज़ार और हाथ साफ़ रखें।",
            "हर मौसम फ़सल बदलें और जहाँ मिले वहाँ रोग-प्रतिरोधी किस्में चुनें।",
            "कोई भी उपचार करने से पहले स्थानीय कृषि विशेषज्ञ से पूछें।",
        ],
        "batch_title": "🗂️ एक साथ कई फ़ोटो की जाँच",
        "batch_intro": "एक बार में {n} तक पत्तियों की फ़ोटो जाँचें। हर नतीजा आपके इतिहास में सहेजा जाता है।",
        "batch_plant": "🌿 इन फ़ोटो में दिख रहा पौधा",
        "batch_upload": "{n} तक फ़ोटो चुनें",
        "batch_limit": "सिर्फ़ पहली {n} फ़ोटो की जाँच होगी।",
        "batch_btn": "🔍 सभी फ़ोटो जाँचें",
        "batch_progress": "फ़ोटो {i} / {n} की जाँच हो रही है...",
        "batch_fail": "{name}: जाँच नहीं हो सकी ({err})",
        "batch_summary": "सारांश",
        "batch_dl": "📥 नतीजे डाउनलोड करें (CSV)",
        "col_status": "स्थिति",
        "col_score": "स्कोर",
        "garden_page_title": "🪴 मेरा बगीचा",
        "garden_intro": "हर पौधे को नाम दें और हर जाँच के बाद एक एंट्री सहेजें। समय के साथ उसकी सेहत देखें।",
        "garden_empty": "आपका बगीचा खाली है। पत्ती की जाँच करें, फिर नतीजे के नीचे 'मेरे बगीचे में सहेजें' दबाएँ।",
        "garden_pick": "पौधा चुनें",
        "garden_checkins": "एंट्री",
        "garden_last": "आख़िरी स्थिति",
        "garden_chart": "समय के साथ सेहत (1 = स्वस्थ अनुमान, 0 = ध्यान चाहिए)",
        "garden_col_date": "एंट्री",
        "garden_col_health": "सेहत",
        "garden_col_status": "नतीजा",
        "garden_ok": "स्वस्थ होने का अनुमान",
        "garden_bad": "ध्यान चाहिए",
        "garden_confirm": "मैं इस पौधे की एंट्री मिटाना चाहता/चाहती हूँ",
        "garden_delete": "🗑️ यह पौधा हटाएँ",
        "garden_deleted": "'{name}' को मेरे बगीचे से हटा दिया गया।",
        "garden_note": "मेरा बगीचा ऐप के साथ ही सहेजा जाता है। मुफ़्त क्लाउड होस्टिंग पर ऐप दोबारा चालू होने पर यह मिट सकता है।",
        "about_title": "ℹ️ PlantHealth AI के बारे में",
        "about_intro": "PlantHealth AI एक प्रशिक्षित डीप लर्निंग मॉडल से फ़ोटो देखकर पत्ती की संभावित स्थिति का अनुमान लगाता है। हर जाँच स्थानीय SQLite डेटाबेस में सहेजी जाती है, ताकि आप बाद में देख सकें।",
        "about_plants": "समर्थित पौधे",
        "about_how": "यह कैसे काम करता है",
        "about_steps": (
            "1. आप पत्ती की फ़ोटो जोड़ते हैं।\n"
            "2. फ़ोटो तैयार करके मॉडल को दी जाती है।\n"
            "3. मॉडल हर जानी-पहचानी श्रेणी को स्कोर देता है।\n"
            "4. सबसे अच्छे मेल देखभाल के सुझावों के साथ दिखते हैं।\n"
            "5. नतीजा आपके इतिहास में सहेजा जाता है।"
        ),
        "about_good": "ध्यान रखने योग्य बातें",
        "about_notes": (
            "- नतीजे अनुमान हैं, पक्की जाँच नहीं।\n"
            "- समर्थित सूची के बाहर के पौधे गलत श्रेणी में आ सकते हैं।\n"
            "- ऊँचा स्कोर सही उत्तर की गारंटी नहीं है।\n"
            "- रोगों के नाम अंग्रेज़ी में दिखाए जाते हैं।\n"
            "- कोई भी उपचार करने से पहले स्थानीय कृषि विशेषज्ञ से पूछें।"
        ),
        "footer": "🌿 PlantHealth AI · शैक्षिक प्रोजेक्ट · एआई के अनुमान गलत हो सकते हैं",
        "report_title": "PLANTHEALTH AI - पत्ती जाँच रिपोर्ट",
    },
}

LANG = "en"


def t(key, lang=None, **kwargs):
    """Look up a translated text. Falls back to English if missing."""
    table = STR.get(lang or LANG, STR["en"])
    value = table.get(key, STR["en"].get(key, key))
    if kwargs and isinstance(value, str):
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return value
    return value


def tr(en, hi):
    """Pick English or Hindi text for the current language."""
    return hi if LANG == "hi" else en


def plant_label(name):
    """Plant name with emoji, in the chosen language."""
    shown = PLANT_HI.get(name, name) if LANG == "hi" else name
    return f"{PLANT_EMOJI.get(name, '🌿')} {shown}"


# ---------------- DISEASE KNOWLEDGE BASE ----------------
# Order matters: the first keyword found in the predicted class is used.
# {plant} is replaced with the plant name found in the prediction.
DISEASES = [
    {
        "key": "apple scab",
        "name": ("Apple scab", "सेब का स्कैब रोग"),
        "title": ("Possible apple scab", "संभावित सेब स्कैब रोग"),
        "about": (
            "Can cause olive-coloured spots on leaves and fruit.",
            "पत्तियों और फलों पर जैतून जैसे रंग के धब्बे पड़ सकते हैं।",
        ),
        "plants": ["Apple"],
    },
    {
        "key": "cedar apple rust",
        "name": ("Cedar apple rust", "सीडर एप्पल रस्ट (जंग रोग)"),
        "title": ("Possible cedar apple rust", "संभावित सीडर एप्पल रस्ट"),
        "about": (
            "May cause yellow-orange leaf spots and related symptoms.",
            "पत्तियों पर पीले-नारंगी धब्बे और उनसे जुड़े लक्षण हो सकते हैं।",
        ),
        "plants": ["Apple"],
    },
    {
        "key": "black rot",
        "name": ("Black rot", "ब्लैक रॉट (काला सड़न रोग)"),
        "title": ("Possible black rot", "संभावित ब्लैक रॉट"),
        "about": (
            "Can cause leaf lesions and other plant tissue damage.",
            "पत्तियों पर घाव और पौधे के दूसरे ऊतकों को नुकसान हो सकता है।",
        ),
        "plants": ["Apple", "Grape"],
    },
    {
        "key": "powdery mildew",
        "name": ("Powdery mildew", "पाउडरी मिल्ड्यू (चूर्णिल आसिता)"),
        "title": ("Possible powdery mildew", "संभावित पाउडरी मिल्ड्यू"),
        "about": (
            "Often appears as white or grey powdery growth on leaves.",
            "अक्सर पत्तियों पर सफ़ेद या धूसर चूर्ण जैसी परत दिखती है।",
        ),
        "plants": ["Cherry", "Squash"],
    },
    {
        "key": "cercospora leaf spot",
        "name": ("Cercospora leaf spot", "सर्कोस्पोरा पत्ती धब्बा रोग"),
        "title": ("Possible Cercospora leaf spot", "संभावित सर्कोस्पोरा पत्ती धब्बा"),
        "about": (
            "May cause leaf spots that expand and damage leaf tissue.",
            "पत्तियों पर धब्बे बन सकते हैं जो फैलकर पत्ती के ऊतकों को नुकसान पहुँचाते हैं।",
        ),
        "plants": ["Corn"],
    },
    {
        "key": "gray leaf spot",
        "name": ("Gray leaf spot", "ग्रे लीफ स्पॉट (धूसर पत्ती धब्बा)"),
        "title": ("Possible gray leaf spot", "संभावित ग्रे लीफ स्पॉट"),
        "about": (
            "Can produce elongated or rectangular lesions on leaves.",
            "पत्तियों पर लंबे या आयताकार घाव बन सकते हैं।",
        ),
        "plants": ["Corn"],
    },
    {
        "key": "common rust",
        "name": ("Common rust", "कॉमन रस्ट (सामान्य जंग रोग)"),
        "title": ("Possible common rust", "संभावित कॉमन रस्ट"),
        "about": (
            "Can produce reddish-brown or rust-coloured pustules on leaves.",
            "पत्तियों पर लाल-भूरे या जंग जैसे रंग के उभार बन सकते हैं।",
        ),
        "plants": ["Corn"],
    },
    {
        "key": "northern leaf blight",
        "name": ("Northern leaf blight", "नॉर्दर्न लीफ ब्लाइट (उत्तरी पत्ती झुलसा)"),
        "title": ("Possible northern leaf blight", "संभावित नॉर्दर्न लीफ ब्लाइट"),
        "about": (
            "May produce long, grey-green or brown lesions on corn leaves.",
            "मक्के की पत्तियों पर लंबे, धूसर-हरे या भूरे घाव बन सकते हैं।",
        ),
        "plants": ["Corn"],
    },
    {
        "key": "esca",
        "name": ("Esca (black measles)", "एस्का रोग"),
        "title": ("Possible grape Esca disease", "संभावित अंगूर का एस्का रोग"),
        "about": (
            "Can cause leaf discoloration and other symptoms in grapevines.",
            "अंगूर की बेलों में पत्तियों का रंग बदलना और दूसरे लक्षण हो सकते हैं।",
        ),
        "plants": ["Grape"],
    },
    {
        "key": "leaf blight",
        "name": ("Leaf blight", "पत्ती झुलसा रोग"),
        "title": ("Possible leaf blight", "संभावित पत्ती झुलसा"),
        "about": (
            "May cause leaf lesions, browning, and damaged leaf tissue.",
            "पत्तियों पर घाव, भूरापन और क्षतिग्रस्त ऊतक हो सकते हैं।",
        ),
        "plants": ["Grape"],
    },
    {
        "key": "haunglongbing",
        "name": ("Citrus greening (Huanglongbing)", "सिट्रस ग्रीनिंग (हुआंगलोंगबिंग)"),
        "title": ("Possible citrus greening", "संभावित सिट्रस ग्रीनिंग"),
        "about": (
            "Can cause uneven leaf mottling, yellowing, and poor fruit development.",
            "पत्तियों पर असमान धब्बेदार पीलापन और फलों का ठीक से न बढ़ना हो सकता है।",
        ),
        "plants": ["Orange"],
    },
    {
        "key": "citrus greening",
        "name": None,  # same condition as above, listed once in the library
        "title": ("Possible citrus greening", "संभावित सिट्रस ग्रीनिंग"),
        "about": (
            "Can cause uneven leaf mottling, yellowing, and poor fruit development.",
            "पत्तियों पर असमान धब्बेदार पीलापन और फलों का ठीक से न बढ़ना हो सकता है।",
        ),
        "plants": ["Orange"],
    },
    {
        "key": "bacterial spot",
        "name": ("Bacterial spot", "जीवाणु धब्बा रोग"),
        "title": ("Possible bacterial spot", "संभावित जीवाणु धब्बा रोग"),
        "about": (
            "May cause dark, water-soaked or necrotic spots on leaves.",
            "पत्तियों पर गहरे, पानी से भीगे या सूखे मृत धब्बे हो सकते हैं।",
        ),
        "plants": ["Peach", "Bell pepper", "Tomato"],
    },
    {
        "key": "early blight",
        "name": ("Early blight", "अर्ली ब्लाइट (अगेती झुलसा)"),
        "title": ("Possible {plant} early blight", "संभावित {plant} अर्ली ब्लाइट"),
        "about": (
            "May cause dark leaf spots, sometimes with concentric rings and surrounding yellowing.",
            "पत्तियों पर गहरे धब्बे हो सकते हैं, कभी-कभी गोल छल्लों और आसपास पीलेपन के साथ।",
        ),
        "plants": ["Potato", "Tomato"],
    },
    {
        "key": "late blight",
        "name": ("Late blight", "लेट ब्लाइट (पछेती झुलसा)"),
        "title": ("Possible {plant} late blight", "संभावित {plant} लेट ब्लाइट"),
        "about": (
            "Can cause rapidly expanding, water-soaked or dark lesions on leaves in favourable conditions.",
            "अनुकूल मौसम में पत्तियों पर तेज़ी से फैलने वाले, पानी से भीगे या गहरे घाव हो सकते हैं।",
        ),
        "plants": ["Potato", "Tomato"],
    },
    {
        "key": "leaf mold",
        "name": ("Leaf mold", "लीफ मोल्ड (पत्ती फफूंद)"),
        "title": ("Possible leaf mold", "संभावित लीफ मोल्ड"),
        "about": (
            "May cause pale yellow patches on upper leaf surfaces and mould growth underneath.",
            "पत्ती की ऊपरी सतह पर हल्के पीले धब्बे और नीचे फफूंद की वृद्धि हो सकती है।",
        ),
        "plants": ["Tomato"],
    },
    {
        "key": "septoria leaf spot",
        "name": ("Septoria leaf spot", "सेप्टोरिया पत्ती धब्बा रोग"),
        "title": ("Possible Septoria leaf spot", "संभावित सेप्टोरिया पत्ती धब्बा"),
        "about": (
            "Often causes multiple small, dark-bordered spots, sometimes with pale centres.",
            "अक्सर कई छोटे, गहरे किनारे वाले धब्बे बनते हैं, जिनका बीच का हिस्सा कभी-कभी हल्का होता है।",
        ),
        "plants": ["Tomato"],
    },
    {
        "key": "spider mites",
        "name": ("Spider mites", "स्पाइडर माइट्स (मकड़ी कीट)"),
        "title": ("Possible spider mite damage", "संभावित स्पाइडर माइट का नुकसान"),
        "about": (
            "Can cause fine stippling, yellowing, and sometimes webbing.",
            "पत्तियों पर बारीक बिंदु, पीलापन और कभी-कभी जाला दिख सकता है।",
        ),
        "plants": ["Tomato"],
    },
    {
        "key": "target spot",
        "name": ("Target spot", "टारगेट स्पॉट"),
        "title": ("Possible target spot", "संभावित टारगेट स्पॉट"),
        "about": (
            "May produce brown lesions with concentric patterns.",
            "भूरे घाव बन सकते हैं जिनमें गोल छल्लों जैसा पैटर्न होता है।",
        ),
        "plants": ["Tomato"],
    },
    {
        "key": "yellow leaf curl virus",
        "name": ("Yellow leaf curl virus", "येलो लीफ कर्ल वायरस (पीला पत्ती मुड़न)"),
        "title": ("Possible yellow leaf curl virus", "संभावित येलो लीफ कर्ल वायरस"),
        "about": (
            "Can cause leaf curling, yellowing, stunted growth, and reduced yield.",
            "पत्तियाँ मुड़ सकती हैं, पीली पड़ सकती हैं, वृद्धि रुक सकती है और उपज घट सकती है।",
        ),
        "plants": ["Tomato"],
    },
    {
        "key": "mosaic virus",
        "name": ("Mosaic virus", "मोज़ेक वायरस"),
        "title": ("Possible mosaic virus", "संभावित मोज़ेक वायरस"),
        "about": (
            "May cause mottled light and dark green areas or distorted leaves.",
            "पत्तियों पर हल्के और गहरे हरे चितकबरे हिस्से या टेढ़ी-मेढ़ी पत्तियाँ हो सकती हैं।",
        ),
        "plants": ["Tomato"],
    },
    {
        "key": "leaf scorch",
        "name": ("Leaf scorch", "पत्ती झुलसन (लीफ स्कॉर्च)"),
        "title": ("Possible leaf scorch", "संभावित लीफ स्कॉर्च"),
        "about": (
            "Can cause browning or scorched-looking areas along leaf edges.",
            "पत्ती के किनारों पर भूरापन या झुलसे हुए हिस्से हो सकते हैं।",
        ),
        "plants": ["Strawberry"],
    },
]


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


def check_image_quality(image):
    """Return simple, friendly warnings about the uploaded photo."""
    warnings = []
    try:
        width, height = image.size
        if min(width, height) < 224:
            warnings.append(t("q_small", w=width, h=height))
        brightness = ImageStat.Stat(image.convert("L")).mean[0]
        if brightness < 50:
            warnings.append(t("q_dark"))
        elif brightness > 215:
            warnings.append(t("q_bright"))
    except Exception:
        pass
    return warnings


def get_disease_guidance(class_name, lang=None):
    """Return guidance for a supported plant condition in the chosen language."""
    lang = lang or LANG
    idx = 1 if lang == "hi" else 0
    name = class_name.lower().replace("_", " ")
    plant = name.split("—")[0].strip() if "—" in name else ""
    plant_title = plant.title() if plant else ("पौधा" if lang == "hi" else "plant")

    general_care = t("g_general", lang)

    if "background" in name:
        return {
            "title": t("g_bg_title", lang),
            "about": t("g_bg_about", lang),
            "care": t("g_bg_care", lang),
        }

    if "healthy" in name:
        return {
            "title": t("g_healthy_title", lang, plant=plant_title),
            "about": t("g_healthy_about", lang),
            "care": t("g_healthy_care", lang),
        }

    for item in DISEASES:
        if item["key"] in name:
            title = item["title"][idx].replace("{plant}", plant_title)
            return {
                "title": title,
                "about": item["about"][idx] + t("g_note", lang),
                "care": general_care,
            }

    return {
        "title": t("g_fallback_title", lang),
        "about": t("g_fallback_about", lang, cls=class_name),
        "care": general_care,
    }


def assess(best_class, confidence, selected_plant):
    """Work out the overall status of a prediction."""
    display_class = format_class_name(best_class)
    predicted_plant = display_class.split("—")[0].strip()
    is_background = "background" in best_class.lower()
    is_healthy = "healthy" in best_class.lower()
    mismatch = (
        not is_background and not plant_matches(selected_plant, predicted_plant)
    )
    low = confidence < CONFIDENCE_THRESHOLD

    if mismatch:
        key, style = "st_mismatch", "bad"
    elif is_background:
        key, style = "st_background", "warn"
    elif low:
        key, style = "st_uncertain", "warn"
    elif is_healthy:
        key, style = "st_healthy", "ok"
    else:
        key, style = "st_disease", "warn"

    return {
        "display_class": display_class,
        "predicted_plant": predicted_plant,
        "is_background": is_background,
        "is_healthy": is_healthy,
        "mismatch": mismatch,
        "low": low,
        "status_key": key,
        "style": style,
        "good": bool(is_healthy and not mismatch and not low),
    }


def build_guidance(res, a, lang=None):
    """Pick the right guidance text for a result."""
    lang = lang or LANG
    if a["mismatch"]:
        return {
            "title": t("g_mm_title", lang),
            "about": t(
                "g_mm_about", lang,
                selected=res["selected_plant"], predicted=a["predicted_plant"],
            ),
            "care": [
                tip.replace("{predicted}", a["predicted_plant"])
                for tip in t("g_mm_care", lang)
            ],
        }
    if a["low"] and not a["is_background"]:
        return {
            "title": t("g_low_title", lang),
            "about": t(
                "g_low_about", lang,
                cls=a["display_class"], score=res["confidence"],
            ),
            "care": t("g_low_care", lang),
        }
    return get_disease_guidance(a["display_class"], lang)


def build_text_report(res, a, guidance):
    """Plain-text report in the chosen language."""
    lines = [
        t("report_title"),
        "=" * 40,
        f"Date and time    : {res['time']}",
        f"Image            : {res['file_name']}",
        f"Selected plant   : {res['selected_plant']}",
        f"Most likely class: {a['display_class']}",
        f"Model score      : {res['confidence']:.2f}%",
        f"Status           : {t(a['status_key'])}",
        "",
        "Top predictions:",
    ]
    for index, item in enumerate(res["top"], start=1):
        lines.append(
            f"  {index}. {format_class_name(item['disease'])} "
            f"({item['confidence']:.2f}%)"
        )
    lines += ["", guidance["title"], guidance["about"], "", t("next_steps") + ":"]
    lines += [f"  - {tip}" for tip in guidance["care"]]
    lines += ["", t("conf_note")]
    return "\n".join(lines)


def build_pdf_report(res, a):
    """Create a designed PDF report (English). Returns bytes or None."""
    if not HAS_PDF:
        return None
    guidance = build_guidance(res, a, "en")
    green = rl_colors.HexColor("#14573F")
    dark = rl_colors.HexColor("#0B3D2E")
    soft = rl_colors.HexColor("#EAF7E4")
    ink = rl_colors.HexColor("#16302A")

    title_style = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=22, textColor=dark, leading=26)
    h_style = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=13, textColor=green, leading=18, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", fontName="Helvetica", fontSize=10.5, textColor=ink, leading=15)
    small = ParagraphStyle("s", fontName="Helvetica-Oblique", fontSize=9, textColor=rl_colors.HexColor("#5B7268"), leading=13)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="PlantHealth AI - Leaf Analysis Report",
    )
    story = [
        Paragraph("PlantHealth AI", title_style),
        Paragraph("Leaf analysis report", body),
        Spacer(1, 10),
    ]

    if res.get("image_bytes"):
        try:
            pil = Image.open(io.BytesIO(res["image_bytes"]))
            w, h = pil.size
            width = 7 * cm
            story += [RLImage(io.BytesIO(res["image_bytes"]), width=width, height=width * h / w), Spacer(1, 8)]
        except Exception:
            pass

    info_rows = [
        ["Date and time", escape(res["time"])],
        ["Image", escape(str(res["file_name"]))],
        ["Selected plant", escape(res["selected_plant"])],
        ["Most likely class", escape(a["display_class"])],
        ["Model score", f"{res['confidence']:.2f}%"],
        ["Status", escape(t(a["status_key"], "en"))],
    ]
    info = Table(
        [[Paragraph(f"<b>{k}</b>", body), Paragraph(v, body)] for k, v in info_rows],
        colWidths=[4.5 * cm, 12.5 * cm],
    )
    info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, rl_colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [info, Paragraph("Top 3 predictions", h_style)]

    rows = [[Paragraph("<b>Class</b>", body), Paragraph("<b>Score</b>", body)]]
    for item in res["top"]:
        rows.append([
            Paragraph(escape(format_class_name(item["disease"])), body),
            Paragraph(f"{item['confidence']:.2f}%", body),
        ])
    top = Table(rows, colWidths=[13 * cm, 4 * cm])
    top.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), soft),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#B7D3BF")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [top, Paragraph(escape(guidance["title"]), h_style), Paragraph(escape(guidance["about"]), body), Paragraph("Recommended next steps", h_style)]
    for tip in guidance["care"]:
        story.append(Paragraph("&bull; " + escape(tip), body))
    story += [
        Spacer(1, 12),
        Paragraph(
            "This is an AI estimate from a leaf photo, not a confirmed diagnosis. "
            "Model confidence is not the same as accuracy. Consult a local "
            "agricultural expert before applying any treatment.",
            small,
        ),
    ]
    doc.build(story)
    return buffer.getvalue()


# ---------------- MY GARDEN (own small SQLite file) ----------------
def garden_connect():
    conn = sqlite3.connect(str(GARDEN_DB))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS garden ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, plant_name TEXT, plant_type TEXT, "
        "checked_at TEXT, prediction TEXT, confidence REAL, healthy INTEGER)"
    )
    return conn


def garden_add(plant_name, plant_type, prediction, confidence, healthy):
    conn = garden_connect()
    try:
        conn.execute(
            "INSERT INTO garden (plant_name, plant_type, checked_at, prediction, "
            "confidence, healthy) VALUES (?, ?, ?, ?, ?, ?)",
            (plant_name, plant_type, datetime.now().strftime("%Y-%m-%d %H:%M"),
             prediction, float(confidence), 1 if healthy else 0),
        )
        conn.commit()
    finally:
        conn.close()


def garden_names():
    conn = garden_connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT plant_name FROM garden ORDER BY plant_name"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def garden_records(plant_name):
    conn = garden_connect()
    try:
        rows = conn.execute(
            "SELECT checked_at, plant_type, prediction, confidence, healthy "
            "FROM garden WHERE plant_name = ? ORDER BY id",
            (plant_name,),
        ).fetchall()
    finally:
        conn.close()
    return rows


def garden_delete(plant_name):
    conn = garden_connect()
    try:
        conn.execute("DELETE FROM garden WHERE plant_name = ?", (plant_name,))
        conn.commit()
    finally:
        conn.close()


# ======================================================================
# PRO TOOLS  (merged from pro_features.py)
# ======================================================================

# ---------- helpers ----------
def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _open(upload):
    try:
        return Image.open(upload).convert("RGB")
    except Exception:
        return None


def _class_conf(item):
    """Read (class, confidence) from a top-prediction item."""
    return item["disease"], float(item["confidence"])


# ---------- 1. leaf colour analysis (no model needed) ----------
def leaf_vitality(image):
    """Estimate leaf colour health from the pixels. Returns a dict or None."""
    small = image.convert("RGB")
    small.thumbnail((256, 256))
    rgb = np.asarray(small)
    hsv = np.asarray(small.convert("HSV")).astype(np.float32)

    hue = hsv[..., 0] * 360.0 / 255.0
    sat = hsv[..., 1]
    val = hsv[..., 2]

    leaf = (sat > 45) & (val > 45)  # ignore white / grey / black background
    total = int(leaf.sum())
    if total < 200:
        return None

    green = leaf & (hue >= 70) & (hue < 170)
    yellow = leaf & (hue >= 35) & (hue < 70) & (val >= 120)
    brown = leaf & ~green & ~yellow & (
        (hue < 35) | (hue >= 330) | ((hue >= 35) & (hue < 70))
    )

    g = float(green.sum()) / total * 100
    y = float(yellow.sum()) / total * 100
    b = float(brown.sum()) / total * 100
    vitality = max(0.0, min(100.0, g + 0.4 * y))

    stress_map = np.full(rgb.shape, 238, dtype=np.uint8)
    stress_map[green] = (46, 160, 90)
    stress_map[yellow] = (245, 200, 30)
    stress_map[brown] = (176, 62, 38)

    return {
        "vitality": vitality,
        "green": g,
        "yellow": y,
        "brown": b,
        "map": Image.fromarray(stress_map),
    }


# ---------- 2. AI Focus Map (works with any predict function) ----------
def _score_for(target, best_class, confidence, top):
    if best_class == target:
        return float(confidence)
    for item in top or []:
        try:
            name, conf = _class_conf(item)
        except Exception:
            continue
        if name == target:
            return conf
    return 0.0  # fell out of the top predictions


def focus_map(image, predict_fn, grid=FOCUS_GRID, progress=None):
    """
    Hide one part of the leaf at a time and see how much the model's
    confidence drops. Big drop = the model relied on that part.
    """
    base = image.convert("RGB")
    base.thumbnail((448, 448))
    best, conf, top = predict_fn(base)
    conf = float(conf)

    w, h = base.size
    mean_color = tuple(int(c) for c in np.asarray(base).reshape(-1, 3).mean(axis=0))
    pad_x, pad_y = int(0.12 * w / grid), int(0.12 * h / grid)

    drops = np.zeros((grid, grid), dtype=np.float32)
    total_cells = grid * grid
    done = 0
    for r in range(grid):
        for c in range(grid):
            x0 = max(0, int(c * w / grid) - pad_x)
            x1 = min(w, int((c + 1) * w / grid) + pad_x)
            y0 = max(0, int(r * h / grid) - pad_y)
            y1 = min(h, int((r + 1) * h / grid) + pad_y)
            patched = base.copy()
            patched.paste(mean_color, (x0, y0, x1, y1))
            b2, c2, t2 = predict_fn(patched)
            drops[r, c] = max(0.0, conf - _score_for(best, b2, float(c2), t2))
            done += 1
            if progress:
                progress(done / total_cells)

    peak = float(drops.max())
    heat = drops / peak if peak > 0 else drops
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), RESAMPLE)
    alpha = (np.asarray(heat_img).astype(np.float32) / 255.0)[..., None] * 0.65

    color = np.zeros((h, w, 3), dtype=np.float32)
    color[..., 0] = 255
    color[..., 1] = 210 * (1 - alpha[..., 0] / 0.65)
    blended = np.asarray(base).astype(np.float32) * (1 - alpha) + color * alpha
    overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    margin = None
    if top and len(top) >= 2:
        try:
            margin = _class_conf(top[0])[1] - _class_conf(top[1])[1]
        except Exception:
            margin = None

    return {
        "best": best,
        "confidence": conf,
        "overlay": overlay,
        "peak_drop": peak,
        "margin": margin,
    }


# ---------- 3. risk advisor (simple rule-based model) ----------
def disease_risk(temp, humidity, rain_days, wet_hours, overhead):
    fungal = 0
    fungal += 35 if humidity >= 80 else 20 if humidity >= 65 else 0
    fungal += 25 if 15 <= temp <= 28 else 0
    fungal += 25 if wet_hours >= 8 else 15 if wet_hours >= 5 else 0
    fungal += 15 if rain_days >= 3 else 7 if rain_days >= 1 else 0
    fungal += 10 if overhead else 0

    bacterial = 0
    bacterial += 30 if humidity >= 75 else 0
    bacterial += 25 if 24 <= temp <= 34 else 0
    bacterial += 25 if rain_days >= 3 else 12 if rain_days >= 1 else 0
    bacterial += 20 if overhead else 0

    pests = 0
    pests += 40 if temp >= 30 else 20 if temp >= 26 else 0
    pests += 30 if humidity <= 50 else 15 if humidity <= 60 else 0
    pests += 20 if rain_days == 0 else 0

    return {
        "fungal": min(100, fungal),
        "bacterial": min(100, bacterial),
        "pests": min(100, pests),
    }


def risk_level(score):
    if score < 35:
        return tr("Low", "कम"), "🟢"
    if score < 65:
        return tr("Medium", "मध्यम"), "🟡"
    return tr("High", "ज़्यादा"), "🔴"


# ---------- 4. care guide data: (sun, water, soil, tip), each (English, Hindi) ----------
CARE = {
    "Apple": (
        ("Full sun, 6-8 hours", "पूरी धूप, 6-8 घंटे"),
        ("Deep watering every week, mulch to keep moisture", "हर हफ़्ते गहरी सिंचाई, नमी के लिए मल्चिंग"),
        ("Well-drained loam, pH 6-7", "अच्छी जल-निकासी वाली दोमट मिट्टी, pH 6-7"),
        ("Prune for air flow and clear fallen leaves.", "हवा के बहाव के लिए छँटाई करें और गिरी पत्तियाँ हटाएँ।"),
    ),
    "Blueberry": (
        ("Full sun", "पूरी धूप"),
        ("Keep soil evenly moist, never waterlogged", "मिट्टी में बराबर नमी रखें, पानी जमा न होने दें"),
        ("Acidic soil, pH 4.5-5.5", "अम्लीय मिट्टी, pH 4.5-5.5"),
        ("Mulch with wood chips or pine bark.", "लकड़ी के चिप्स या चीड़ की छाल से मल्चिंग करें।"),
    ),
    "Cherry": (
        ("Full sun", "पूरी धूप"),
        ("Deep but infrequent watering", "गहरा पर कम बार पानी"),
        ("Well-drained soil, pH 6-7", "अच्छी जल-निकासी वाली मिट्टी, pH 6-7"),
        ("Avoid waterlogging and prune for airflow.", "पानी जमा होने से बचें और हवा के लिए छँटाई करें।"),
    ),
    "Corn": (
        ("Full sun", "पूरी धूप"),
        ("Regular water, especially at tasseling and silking", "नियमित पानी, खासकर बाली और रेशे निकलते समय"),
        ("Fertile, well-drained soil rich in nitrogen", "उपजाऊ, अच्छी जल-निकासी वाली नाइट्रोजन-समृद्ध मिट्टी"),
        ("Plant in blocks, not one long row, for better pollination.", "बेहतर परागण के लिए एक लंबी कतार की जगह ब्लॉक में बोएँ।"),
    ),
    "Grape": (
        ("Full sun", "पूरी धूप"),
        ("Deep watering, less once established", "गहरा पानी, पौधा जमने पर कम"),
        ("Well-drained soil, pH 5.5-6.5", "अच्छी जल-निकासी वाली मिट्टी, pH 5.5-6.5"),
        ("Train on a trellis and prune for air flow.", "जाली पर चढ़ाएँ और हवा के लिए छँटाई करें।"),
    ),
    "Orange": (
        ("Full sun", "पूरी धूप"),
        ("Deep watering; let the top soil dry a little between", "गहरा पानी; बीच में ऊपरी मिट्टी थोड़ी सूखने दें"),
        ("Well-drained, slightly acidic soil, pH 6-7", "अच्छी जल-निकासी वाली हल्की अम्लीय मिट्टी, pH 6-7"),
        ("Watch for small sap-sucking insects on new leaves.", "नई पत्तियों पर रस चूसने वाले छोटे कीटों पर नज़र रखें।"),
    ),
    "Peach": (
        ("Full sun", "पूरी धूप"),
        ("Deep, regular watering", "गहरा और नियमित पानी"),
        ("Well-drained loam, pH 6-6.5", "अच्छी जल-निकासी वाली दोमट मिट्टी, pH 6-6.5"),
        ("Prune to an open centre and thin the fruit.", "बीच से खुला आकार रखकर छँटाई करें और फल पतले करें।"),
    ),
    "Bell pepper": (
        ("Full sun, 6-8 hours", "पूरी धूप, 6-8 घंटे"),
        ("Steady moisture, water at the base", "लगातार नमी, जड़ के पास पानी दें"),
        ("Fertile, well-drained soil, pH 6-6.8", "उपजाऊ, अच्छी जल-निकासी वाली मिट्टी, pH 6-6.8"),
        ("Mulch and avoid wetting the leaves.", "मल्चिंग करें और पत्तियों को गीला न करें।"),
    ),
    "Potato": (
        ("Full sun", "पूरी धूप"),
        ("Even moisture, avoid waterlogging", "बराबर नमी, पानी जमा न होने दें"),
        ("Loose, well-drained, slightly acidic soil, pH 5-6.5", "ढीली, अच्छी जल-निकासी वाली हल्की अम्लीय मिट्टी, pH 5-6.5"),
        ("Use healthy seed tubers, hill up soil and rotate crops.", "स्वस्थ बीज कंद लें, मिट्टी चढ़ाएँ और फ़सल बदलें।"),
    ),
    "Raspberry": (
        ("Full sun to light shade", "पूरी धूप से हल्की छाँव"),
        ("Regular moisture, mulch the base", "नियमित नमी, जड़ पर मल्चिंग करें"),
        ("Well-drained soil, pH 5.5-6.5", "अच्छी जल-निकासी वाली मिट्टी, pH 5.5-6.5"),
        ("Remove old canes and keep plants spaced.", "पुरानी टहनियाँ हटाएँ और पौधों में दूरी रखें।"),
    ),
    "Soybean": (
        ("Full sun", "पूरी धूप"),
        ("Moderate water; most important at flowering and pod fill", "सामान्य पानी; फूल और फली भरते समय सबसे ज़रूरी"),
        ("Well-drained loam, pH 6-7", "अच्छी जल-निकासी वाली दोमट मिट्टी, pH 6-7"),
        ("Rotate with cereal crops to break disease cycles.", "रोग चक्र तोड़ने के लिए अनाज वाली फ़सलों से अदला-बदली करें।"),
    ),
    "Squash": (
        ("Full sun", "पूरी धूप"),
        ("Deep, regular watering at the base", "जड़ के पास गहरा और नियमित पानी"),
        ("Rich, well-drained soil, pH 6-6.8", "उपजाऊ, अच्छी जल-निकासी वाली मिट्टी, pH 6-6.8"),
        ("Give plants room; wet leaves invite mildew.", "पौधों को जगह दें; गीली पत्तियों पर फफूंद लगती है।"),
    ),
    "Strawberry": (
        ("Full sun", "पूरी धूप"),
        ("Regular, light watering; drip is ideal", "नियमित हल्का पानी; ड्रिप सबसे अच्छी"),
        ("Well-drained, slightly acidic soil, pH 5.5-6.5", "अच्छी जल-निकासी वाली हल्की अम्लीय मिट्टी, pH 5.5-6.5"),
        ("Remove old or spotted leaves after harvest.", "फ़सल के बाद पुरानी या धब्बेदार पत्तियाँ हटाएँ।"),
    ),
    "Tomato": (
        ("Full sun, 6-8 hours", "पूरी धूप, 6-8 घंटे"),
        ("Deep, regular watering at the base", "जड़ के पास गहरा और नियमित पानी"),
        ("Rich, well-drained soil, pH 6-6.8", "उपजाऊ, अच्छी जल-निकासी वाली मिट्टी, pH 6-6.8"),
        ("Stake plants, mulch, and rotate where you plant each year.", "पौधों को सहारा दें, मल्चिंग करें और हर साल जगह बदलें।"),
    ),
}

TIPS = [
    ("Water the soil, not the leaves. Wet leaves invite disease.", "पत्तियों पर नहीं, मिट्टी में पानी दें। गीली पत्तियों पर रोग लगते हैं।"),
    ("Check the underside of leaves. Many pests hide there.", "पत्तियों के नीचे की तरफ़ देखें। कई कीट वहीं छिपते हैं।"),
    ("Morning is the best time to water. Leaves dry quickly in the sun.", "सुबह पानी देना सबसे अच्छा है। धूप में पत्तियाँ जल्दी सूख जाती हैं।"),
    ("Clean your tools between plants to avoid spreading disease.", "रोग फैलने से बचाने के लिए पौधों के बीच औज़ार साफ़ करें।"),
    ("Rotating crops each season breaks the life cycle of many diseases.", "हर मौसम फ़सल बदलने से कई रोगों का चक्र टूटता है।"),
    ("Take photos of one leaf every week to spot slow changes early.", "धीमे बदलाव जल्दी पकड़ने के लिए हर हफ़्ते एक पत्ती की फ़ोटो लें।"),
    ("Good spacing means good air flow, and air flow means fewer fungal problems.", "सही दूरी से हवा चलती है, और हवा चलने से फफूंद की समस्या कम होती है।"),
]

QUIZ = [
    (
        ("What is the best way to water a plant to reduce leaf diseases?", "पत्तियों के रोग कम करने के लिए पौधे को पानी देने का सबसे अच्छा तरीका क्या है?"),
        [("Water at the base of the plant", "पौधे की जड़ के पास पानी दें"), ("Spray water over the leaves at night", "रात में पत्तियों पर पानी छिड़कें"), ("Water only when leaves wilt completely", "पत्तियाँ पूरी तरह मुरझाने पर ही पानी दें")],
        0,
        ("Wet leaves, especially overnight, help fungi and bacteria spread.", "गीली पत्तियाँ, खासकर रात भर, फफूंद और जीवाणु को फैलने में मदद करती हैं।"),
    ),
    (
        ("Does a high model score guarantee the diagnosis is correct?", "क्या ऊँचा मॉडल स्कोर सही निदान की गारंटी देता है?"),
        [("Yes, always", "हाँ, हमेशा"), ("No, predictions can still be wrong", "नहीं, अनुमान फिर भी गलत हो सकते हैं"), ("Only for tomato", "सिर्फ़ टमाटर के लिए")],
        1,
        ("Confidence is not accuracy. Unclear photos or unsupported plants can still get a high score.", "भरोसा और सटीकता अलग हैं। धुंधली फ़ोटो या असमर्थित पौधे को भी ऊँचा स्कोर मिल सकता है।"),
    ),
    (
        ("Powdery mildew usually looks like...", "पाउडरी मिल्ड्यू आमतौर पर कैसा दिखता है?"),
        [("White or grey powder on leaves", "पत्तियों पर सफ़ेद या धूसर चूर्ण"), ("Holes made by caterpillars", "इल्लियों से बने छेद"), ("Blue spots on stems", "तने पर नीले धब्बे")],
        0,
        ("It forms a powdery white or grey coating on leaf surfaces.", "यह पत्ती की सतह पर सफ़ेद या धूसर चूर्ण जैसी परत बनाता है।"),
    ),
    (
        ("Common rust on corn appears as...", "मक्के में कॉमन रस्ट कैसा दिखता है?"),
        [("Reddish-brown pustules on leaves", "पत्तियों पर लाल-भूरे उभार"), ("White cotton-like webs", "सफ़ेद रुई जैसे जाले"), ("Purple roots", "बैंगनी जड़ें")],
        0,
        ("Rust-coloured raised pustules on both leaf surfaces are the typical sign.", "पत्ती की दोनों सतहों पर जंग जैसे रंग के उभार इसका आम लक्षण हैं।"),
    ),
    (
        ("Is brinjal (eggplant) supported by this app?", "क्या यह ऐप बैंगन को पहचान सकता है?"),
        [("Yes, fully", "हाँ, पूरी तरह"), ("No, it is not in the 14 supported plants", "नहीं, यह 14 समर्थित पौधों में नहीं है"), ("Only in Hindi mode", "सिर्फ़ हिन्दी मोड में")],
        1,
        ("An unsupported plant may be pushed into a wrong class, even with a high score.", "असमर्थित पौधा ऊँचे स्कोर के साथ भी गलत श्रेणी में जा सकता है।"),
    ),
    (
        ("Why does crop rotation help?", "फ़सल चक्र क्यों फ़ायदेमंद है?"),
        [("It breaks the life cycle of soil-borne diseases and pests", "यह मिट्टी के रोगों और कीटों का जीवन चक्र तोड़ता है"), ("It makes leaves greener instantly", "इससे पत्तियाँ तुरंत हरी हो जाती हैं"), ("It removes the need for water", "इससे पानी की ज़रूरत खत्म हो जाती है")],
        0,
        ("Many pathogens survive in soil or residue. Changing the crop starves them.", "कई रोगाणु मिट्टी या अवशेषों में बचे रहते हैं। फ़सल बदलने से वे पनप नहीं पाते।"),
    ),
]


# ---------- Pro sub-tab: Leaf Insights ----------
def _vitality_message(score):
    if score >= 75:
        return "success", tr("Vibrant colour. The leaf looks mostly green.", "जीवंत रंग। पत्ती ज़्यादातर हरी दिख रही है।")
    if score >= 45:
        return "warning", tr("Some colour stress: noticeable yellow or brown areas.", "कुछ रंग-तनाव: पीले या भूरे हिस्से दिख रहे हैं।")
    return "error", tr("Strong colour stress: much of the leaf is yellow or brown.", "ज़्यादा रंग-तनाव: पत्ती का बड़ा हिस्सा पीला या भूरा है।")


def _tab_insights():
    st.markdown("### " + tr("🧠 Leaf Insights", "🧠 पत्ती की गहरी जाँच"))
    st.write(tr(
        "Two extra checks on one photo: a colour-based Vitality Score, and an AI Focus Map that shows which part of the leaf the model looked at.",
        "एक फ़ोटो पर दो अतिरिक्त जाँच: रंग पर आधारित जीवन-शक्ति स्कोर, और एआई फ़ोकस मैप जो दिखाता है कि मॉडल ने पत्ती का कौन-सा हिस्सा देखा।",
    ))
    up = st.file_uploader(
        tr("Upload a leaf photo", "पत्ती की फ़ोटो अपलोड करें"),
        type=["jpg", "jpeg", "png"], key="pro_ins_upload",
    )
    if up is None:
        st.info(tr("Upload a photo to begin.", "शुरू करने के लिए फ़ोटो अपलोड करें।"))
        return
    image = _open(up)
    if image is None:
        st.error(tr("This image could not be opened.", "यह फ़ोटो खोली नहीं जा सकी।"))
        return

    vit = leaf_vitality(image)
    c1, c2 = st.columns(2)
    with c1:
        st.image(image, caption=tr("Your photo", "आपकी फ़ोटो"))
    with c2:
        if vit:
            st.image(vit["map"], caption=tr("Colour stress map", "रंग-तनाव मानचित्र"))
            st.caption(tr("🟩 green = healthy colour   🟨 yellow = fading   🟥 brown = damaged or dry",
                          "🟩 हरा = स्वस्थ रंग   🟨 पीला = फीका पड़ता   🟥 भूरा = क्षतिग्रस्त या सूखा"))

    if vit is None:
        st.warning(tr("No clear leaf colours found. Try a closer photo with a plain background.",
                      "साफ़ पत्ती के रंग नहीं मिले। सादी पृष्ठभूमि पर नज़दीक से फ़ोटो लें।"))
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(tr("Vitality Score", "जीवन-शक्ति स्कोर"), f"{vit['vitality']:.0f}/100")
        m2.metric(tr("Green", "हरा"), f"{vit['green']:.0f}%")
        m3.metric(tr("Yellow", "पीला"), f"{vit['yellow']:.0f}%")
        m4.metric(tr("Brown", "भूरा"), f"{vit['brown']:.0f}%")
        st.progress(min(max(vit["vitality"] / 100, 0.0), 1.0))
        kind, msg = _vitality_message(vit["vitality"])
        getattr(st, kind)(msg)
        st.caption(tr(
            "This score only measures colour. Some healthy leaves are naturally yellow or red, so it is a hint, not a diagnosis.",
            "यह स्कोर सिर्फ़ रंग मापता है। कुछ स्वस्थ पत्तियाँ स्वाभाविक रूप से पीली या लाल होती हैं, इसलिए यह संकेत है, निदान नहीं।",
        ))
        st.download_button(
            tr("📥 Download stress map (PNG)", "📥 तनाव मानचित्र डाउनलोड करें (PNG)"),
            data=_png(vit["map"]), file_name="leaf_stress_map.png", mime="image/png",
            key="pro_dl_stress",
        )

    st.divider()
    st.markdown("#### " + tr("🔥 AI Focus Map", "🔥 एआई फ़ोकस मैप"))
    st.write(tr(
        f"The app hides {FOCUS_GRID * FOCUS_GRID} parts of the leaf one by one and checks how much the model's confidence drops. Red areas mattered most to the model. This takes a little while.",
        f"ऐप पत्ती के {FOCUS_GRID * FOCUS_GRID} हिस्सों को एक-एक करके छिपाता है और देखता है कि मॉडल का भरोसा कितना घटता है। लाल हिस्से मॉडल के लिए सबसे ज़रूरी थे। इसमें थोड़ा समय लगता है।",
    ))
    sig = (getattr(up, "name", ""), getattr(up, "size", 0))
    if st.button(tr("Run Focus Map", "फ़ोकस मैप चलाएँ"), type="primary", key="pro_focus_btn"):
        bar = st.progress(0.0)
        try:
            result = focus_map(image, predict_plant_disease, progress=bar.progress)
            st.session_state["pro_focus"] = {"sig": sig, "result": result}
        except Exception as exc:
            st.error(tr(f"Could not build the Focus Map: {exc}", f"फ़ोकस मैप नहीं बन सका: {exc}"))
        bar.empty()

    saved = st.session_state.get("pro_focus")
    if saved and saved["sig"] == sig:
        res = saved["result"]
        f1, f2 = st.columns(2)
        with f1:
            st.image(res["overlay"], caption=tr("Where the model looked", "मॉडल ने कहाँ देखा"))
        with f2:
            st.markdown(f"**{tr('Model prediction', 'मॉडल का अनुमान')}**")
            st.write(format_class_name(res["best"]))
            st.metric(tr("Model score", "मॉडल स्कोर"), f"{res['confidence']:.2f}%")
            if res["margin"] is not None:
                m = res["margin"]
                if m >= 40:
                    label = tr("Clear winner", "साफ़ विजेता")
                elif m >= 15:
                    label = tr("Moderate gap", "मध्यम अंतर")
                else:
                    label = tr("Close call", "कड़ा मुक़ाबला")
                st.metric(tr("Gap to 2nd guess", "दूसरे अनुमान से अंतर"), f"{m:.1f} pts", help=label)
                st.caption(f"{tr('Reliability hint', 'भरोसे का संकेत')}: {label}")
        if res["peak_drop"] <= 0:
            st.info(tr("Hiding single parts did not change the result, so no single area stood out.",
                       "किसी एक हिस्से को छिपाने से नतीजा नहीं बदला, इसलिए कोई एक हिस्सा अलग नहीं दिखा।"))
        st.caption(tr(
            "The Focus Map is an approximation built from the model's top predictions. It shows what influenced the model, not proof of where the disease is.",
            "फ़ोकस मैप मॉडल के शीर्ष अनुमानों से बना एक अनुमान है। यह दिखाता है कि मॉडल पर किसका असर पड़ा, यह नहीं कि रोग ठीक कहाँ है।",
        ))
        st.download_button(
            tr("📥 Download Focus Map (PNG)", "📥 फ़ोकस मैप डाउनलोड करें (PNG)"),
            data=_png(res["overlay"]), file_name="leaf_focus_map.png", mime="image/png",
            key="pro_dl_focus",
        )


# ---------- Pro sub-tab: Compare ----------
def _analyse_for_compare(image):
    best, conf, _top = predict_plant_disease(image)
    return {
        "image": image,
        "best": best,
        "conf": float(conf),
        "healthy": "healthy" in str(best).lower(),
        "vit": leaf_vitality(image),
    }


def _tab_compare():
    st.markdown("### " + tr("⚖️ Compare Two Leaves", "⚖️ दो पत्तियों की तुलना"))
    st.write(tr(
        "Upload a 'before' and an 'after' photo of the same plant to see if it is improving.",
        "एक ही पौधे की 'पहले' और 'बाद' की फ़ोटो अपलोड करें और देखें कि सुधार हो रहा है या नहीं।",
    ))
    col_a, col_b = st.columns(2)
    with col_a:
        up_a = st.file_uploader(tr("Before", "पहले"), type=["jpg", "jpeg", "png"], key="pro_cmp_a")
    with col_b:
        up_b = st.file_uploader(tr("After", "बाद"), type=["jpg", "jpeg", "png"], key="pro_cmp_b")

    ready = up_a is not None and up_b is not None
    if st.button(tr("Compare now", "अभी तुलना करें"), type="primary", disabled=not ready, key="pro_cmp_btn"):
        img_a, img_b = _open(up_a), _open(up_b)
        if img_a is None or img_b is None:
            st.error(tr("One of the images could not be opened.", "एक फ़ोटो खोली नहीं जा सकी।"))
        else:
            try:
                with st.spinner(tr("Comparing...", "तुलना हो रही है...")):
                    st.session_state["pro_compare"] = {
                        "a": _analyse_for_compare(img_a),
                        "b": _analyse_for_compare(img_b),
                    }
            except Exception as exc:
                st.error(tr(f"Could not compare: {exc}", f"तुलना नहीं हो सकी: {exc}"))

    data = st.session_state.get("pro_compare")
    if not data:
        return
    a, b = data["a"], data["b"]
    ca, cb = st.columns(2)
    for col, label, item in ((ca, tr("Before", "पहले"), a), (cb, tr("After", "बाद"), b)):
        with col:
            st.image(item["image"], caption=label)
            st.write(f"**{format_class_name(item['best'])}**")
            st.caption(f"{tr('Model score', 'मॉडल स्कोर')}: {item['conf']:.2f}%")
            if item["vit"]:
                st.caption(f"{tr('Vitality Score', 'जीवन-शक्ति स्कोर')}: {item['vit']['vitality']:.0f}/100")

    delta = None
    if a["vit"] and b["vit"]:
        delta = b["vit"]["vitality"] - a["vit"]["vitality"]

    if b["healthy"] and not a["healthy"]:
        st.success(tr("Improving: the model moved from a disease class to a healthy class.",
                      "सुधार: मॉडल रोग श्रेणी से स्वस्थ श्रेणी पर आ गया।"))
    elif a["healthy"] and not b["healthy"]:
        st.error(tr("Getting worse: the model moved from a healthy class to a disease class.",
                    "बिगड़ रहा है: मॉडल स्वस्थ श्रेणी से रोग श्रेणी पर आ गया।"))
    elif delta is not None and delta >= 8:
        st.success(tr(f"Colour improved by {delta:.0f} points.", f"रंग में {delta:.0f} अंक का सुधार हुआ।"))
    elif delta is not None and delta <= -8:
        st.warning(tr(f"Colour dropped by {abs(delta):.0f} points.", f"रंग में {abs(delta):.0f} अंक की गिरावट आई।"))
    else:
        st.info(tr("No big change between the two photos.", "दोनों फ़ोटो में कोई बड़ा बदलाव नहीं है।"))
    st.caption(tr("Use similar lighting and distance in both photos for a fair comparison.",
                  "सही तुलना के लिए दोनों फ़ोटो में एक जैसी रोशनी और दूरी रखें।"))


# ---------- Pro sub-tab: Risk advisor ----------
def _tab_risk():
    st.markdown("### " + tr("🌦️ Disease Risk Advisor", "🌦️ रोग जोखिम सलाहकार"))
    st.write(tr(
        "Enter your field conditions to see how favourable they are for common plant problems. This is a simple rule-based guide, not a weather forecast.",
        "अपने खेत की स्थिति डालें और देखें कि वह आम पौधा-समस्याओं के लिए कितनी अनुकूल है। यह सरल नियमों पर आधारित गाइड है, मौसम पूर्वानुमान नहीं।",
    ))

    plant = st.selectbox(
        tr("Plant", "पौधा"), PLANT_OPTIONS,
        index=PLANT_OPTIONS.index("Tomato"), format_func=plant_label, key="pro_risk_plant",
    )
    c1, c2 = st.columns(2)
    with c1:
        temp = st.slider(tr("Average temperature (°C)", "औसत तापमान (°C)"), 5, 45, 27, key="pro_risk_temp")
        humidity = st.slider(tr("Humidity (%)", "आर्द्रता (%)"), 10, 100, 70, key="pro_risk_hum")
    with c2:
        rain = st.slider(tr("Rainy days in the last 7 days", "पिछले 7 दिनों में बारिश के दिन"), 0, 7, 2, key="pro_risk_rain")
        wet = st.slider(tr("Hours leaves stay wet each night", "हर रात पत्तियाँ कितने घंटे गीली रहती हैं"), 0, 14, 4, key="pro_risk_wet")
    irrigation = st.radio(
        tr("How do you water?", "आप पानी कैसे देते हैं?"), ["base", "overhead"], horizontal=True,
        format_func=lambda x: tr("At the base / drip", "जड़ के पास / ड्रिप") if x == "base" else tr("Sprinkler / over the leaves", "स्प्रिंकलर / पत्तियों के ऊपर से"),
        key="pro_risk_irr",
    )

    risks = disease_risk(temp, humidity, rain, wet, irrigation == "overhead")
    rows = [
        ("fungal", tr("Fungal diseases (blight, mildew, leaf spots)", "फफूंद रोग (झुलसा, मिल्ड्यू, पत्ती धब्बे)")),
        ("bacterial", tr("Bacterial diseases (bacterial spot)", "जीवाणु रोग (जीवाणु धब्बा)")),
        ("pests", tr("Pests and virus-spreading insects (mites, whitefly)", "कीट और वायरस फैलाने वाले कीड़े (माइट्स, सफ़ेद मक्खी)")),
    ]
    st.markdown("#### " + tr("Risk levels", "जोखिम का स्तर"))
    for key, label in rows:
        level, dot = risk_level(risks[key])
        st.write(f"**{label}**: {dot} {level} ({risks[key]}/100)")
        st.progress(risks[key] / 100)

    watch = [d for d in DISEASES if d.get("name") and plant in d["plants"]]
    idx = 1 if LANG == "hi" else 0
    if watch:
        plant_shown = PLANT_HI.get(plant, plant)
        st.markdown("#### " + tr(f"Conditions to watch for on {plant}", f"{plant_shown} पर ध्यान रखने योग्य स्थितियाँ"))
        st.markdown("\n".join(f"- {d['name'][idx]}" for d in watch))

    advice = []
    if risks["fungal"] >= 35:
        advice.append(tr("Water early in the morning and keep leaves dry.", "सुबह जल्दी पानी दें और पत्तियाँ सूखी रखें।"))
        advice.append(tr("Space plants out and prune crowded growth for air flow.", "हवा के बहाव के लिए पौधों में दूरी रखें और घनी बढ़त छाँटें।"))
    if risks["bacterial"] >= 35:
        advice.append(tr("Avoid working among wet plants, and clean your tools.", "गीले पौधों के बीच काम न करें और औज़ार साफ़ रखें।"))
    if risks["pests"] >= 35:
        advice.append(tr("Check the underside of leaves for mites and whitefly.", "माइट्स और सफ़ेद मक्खी के लिए पत्तियों के नीचे देखें।"))
    if not advice:
        advice.append(tr("Conditions look fairly calm. Keep inspecting leaves weekly.", "स्थिति अपेक्षाकृत शांत है। हर हफ़्ते पत्तियाँ जाँचते रहें।"))
    st.markdown("#### " + tr("Suggested actions", "सुझाए गए कदम"))
    st.markdown("\n".join(f"- {x}" for x in advice))
    st.caption(tr("This is a general guide. Ask a local agricultural expert before using any treatment.",
                  "यह सामान्य गाइड है। कोई भी उपचार करने से पहले स्थानीय कृषि विशेषज्ञ से पूछें।"))


# ---------- Pro sub-tab: Care guide ----------
def _tab_care():
    st.markdown("### " + tr("🌿 Plant Care Guide", "🌿 पौधा देखभाल गाइड"))
    idx = 1 if LANG == "hi" else 0
    search = st.text_input(tr("🔎 Search a plant", "🔎 पौधा खोजें"), key="pro_care_search").strip().lower()
    labels = (tr("☀️ Sunlight", "☀️ धूप"), tr("💧 Water", "💧 पानी"), tr("🪴 Soil", "🪴 मिट्टी"), tr("💡 Tip", "💡 सुझाव"))
    shown_count = 0
    for p in PLANT_OPTIONS:
        haystack = f"{p} {PLANT_HI.get(p, '')}".lower()
        if search and search not in haystack:
            continue
        shown_count += 1
        with st.expander(plant_label(p)):
            for label, pair in zip(labels, CARE[p]):
                st.markdown(f"**{label}:** {pair[idx]}")
    if shown_count == 0:
        st.info(tr("No plant matches your search.", "आपकी खोज से कोई पौधा नहीं मिला।"))
    st.caption(tr("General guidance. Local climate and soil can change what works best.",
                  "सामान्य मार्गदर्शन। स्थानीय जलवायु और मिट्टी के हिसाब से बेहतर तरीका बदल सकता है।"))


# ---------- Pro sub-tab: Assistant (offline, rule based) ----------
STOP_WORDS = {"what", "about", "tell", "does", "have", "with", "this", "that", "from", "plant",
              "plants", "disease", "diseases", "leaf", "leaves", "spot", "virus", "how", "can",
              "the", "for", "and", "are", "is", "my", "me", "treat", "cure", "symptoms", "when",
              "why", "care", "prevent"}


def _disease_reply(d):
    idx = 1 if LANG == "hi" else 0
    plants = ", ".join(PLANT_HI.get(p, p) if LANG == "hi" else p for p in d["plants"])
    prevent = tr(
        "Water at the base, keep good spacing, remove badly affected leaves and ask a local agricultural expert before any treatment.",
        "जड़ के पास पानी दें, पौधों में दूरी रखें, बहुत प्रभावित पत्तियाँ हटाएँ और कोई भी उपचार करने से पहले स्थानीय कृषि विशेषज्ञ से पूछें।",
    )
    return (f"**{d['name'][idx]}**\n\n"
            f"{tr('Commonly seen on', 'आमतौर पर इन पर दिखता है')}: {plants}\n\n"
            f"{tr('What it may look like', 'यह कैसा दिख सकता है')}: {d['about'][idx]}\n\n"
            f"{tr('What helps', 'क्या मदद करता है')}: {prevent}")


def assistant_answer(question):
    q = question.strip()
    ql = q.lower()
    idx = 1 if LANG == "hi" else 0
    named = [d for d in DISEASES if d.get("name")]

    if any(w in ql for w in ("brinjal", "eggplant", "baingan")) or "बैंगन" in q:
        return tr("Brinjal (eggplant) is not supported by this app. The model may put it in a wrong class even with a high score, so please do not rely on it for brinjal.",
                  "इस ऐप में बैंगन समर्थित नहीं है। मॉडल इसे ऊँचे स्कोर के साथ भी गलत श्रेणी में रख सकता है, इसलिए बैंगन के लिए इस पर भरोसा न करें।")

    if ql.strip(" !.") in ("hi", "hello", "hey", "namaste") or "नमस्ते" in q:
        return tr("Hello! Ask me about a plant disease, plant care, or how to use the app.",
                  "नमस्ते! मुझसे किसी पौधा-रोग, पौधे की देखभाल या ऐप के उपयोग के बारे में पूछें।")

    # 1. exact disease name (the longest matching name wins)
    best_match, best_len = None, 0
    for d in named:
        en = d["name"][0].lower().split("(")[0].strip()
        hi = d["name"][1].split("(")[0].strip()
        if en in ql and len(en) > best_len:
            best_match, best_len = d, len(en)
        elif hi in q and len(hi) > best_len:
            best_match, best_len = d, len(hi)
    if best_match:
        return _disease_reply(best_match)

    # 2. plant mentioned
    plant = None
    for p in PLANT_OPTIONS:
        if p.lower() in ql or PLANT_HI.get(p, "@@") in q:
            plant = p
            break
    care_words = ("water", "sun", "soil", "care", "grow", "light", "पानी", "धूप", "मिट्टी", "देखभाल", "उगा")
    if plant and any(w in ql or w in q for w in care_words):
        sun, water, soil, tip = (pair[idx] for pair in CARE[plant])
        name = PLANT_HI.get(plant, plant) if LANG == "hi" else plant
        return f"**{name}**\n\n- ☀️ {sun}\n- 💧 {water}\n- 🪴 {soil}\n- 💡 {tip}"
    if plant:
        found = [d for d in named if plant in d["plants"]]
        name = PLANT_HI.get(plant, plant) if LANG == "hi" else plant
        if found:
            lst = "\n".join(f"- {d['name'][idx]}" for d in found)
            return tr(f"Conditions this app can recognise on {name}:\n\n{lst}\n\nAsk about any of them for details.",
                      f"इस ऐप द्वारा {name} पर पहचानी जा सकने वाली स्थितियाँ:\n\n{lst}\n\nविवरण के लिए किसी का भी नाम पूछें।")

    # 3. partial word match (e.g. only "blight")
    words = [w.strip("?,.!") for w in ql.split()]
    words = [w for w in words if len(w) >= 5 and w not in STOP_WORDS]
    matches = [d for d in named if any(w in d["name"][0].lower() for w in words)]
    if len(matches) == 1:
        return _disease_reply(matches[0])
    if len(matches) > 1:
        lst = "\n".join(f"- {d['name'][idx]}" for d in matches)
        return tr(f"I found several matches. Which one do you mean?\n\n{lst}",
                  f"कई मिलते-जुलते नतीजे मिले। आपका मतलब कौन-सा है?\n\n{lst}")

    # 4. general topics
    if any(w in ql for w in ("accuracy", "confidence", "score", "sure", "reliable")) or "भरोसा" in q or "स्कोर" in q:
        return tr("The model score shows how strongly the model favours its top answer. It is not the same as accuracy. Blurry photos or unsupported plants can still get a high score, so treat results as estimates.",
                  "मॉडल स्कोर बताता है कि मॉडल अपने शीर्ष उत्तर को कितना पसंद करता है। यह सटीकता नहीं है। धुंधली फ़ोटो या असमर्थित पौधों को भी ऊँचा स्कोर मिल सकता है, इसलिए नतीजे अनुमान मानें।")
    if any(w in ql for w in ("organic", "pesticide", "spray", "chemical", "fungicide", "medicine", "treatment", "medicin")) or "दवा" in q or "छिड़काव" in q:
        return tr("This app does not recommend specific chemicals. Start with safe steps: remove badly affected leaves, improve air flow, water at the base. Then ask a local agricultural expert what is suitable for your crop and area.",
                  "यह ऐप किसी खास रसायन की सलाह नहीं देता। पहले सुरक्षित कदम उठाएँ: बहुत प्रभावित पत्तियाँ हटाएँ, हवा का बहाव सुधारें, जड़ के पास पानी दें। फिर स्थानीय कृषि विशेषज्ञ से पूछें कि आपकी फ़सल और क्षेत्र के लिए क्या सही है।")
    if any(w in ql for w in ("prevent", "avoid", "protect")) or "बचाव" in q:
        return tr("General prevention: water at the base, keep plants spaced, clean tools, remove badly affected leaves, rotate crops and choose resistant varieties.",
                  "सामान्य बचाव: जड़ के पास पानी दें, पौधों में दूरी रखें, औज़ार साफ़ रखें, बहुत प्रभावित पत्तियाँ हटाएँ, फ़सल बदलें और रोग-प्रतिरोधी किस्में चुनें।")
    if any(w in ql for w in ("use", "start", "help", "upload", "photo")) or "कैसे" in q:
        return tr("Open the Analyze tab, choose the plant, upload or capture one clear leaf photo, and click Analyze Leaf. Use Batch for several photos and My Garden to track a plant over time.",
                  "जाँच टैब खोलें, पौधा चुनें, एक साफ़ पत्ती की फ़ोटो अपलोड करें या खींचें और 'पत्ती की जाँच करें' दबाएँ। कई फ़ोटो के लिए 'एक साथ जाँच' और समय के साथ पौधा देखने के लिए 'मेरा बगीचा' इस्तेमाल करें।")

    return tr("I do not have an answer for that yet. Try asking about a disease (for example 'late blight'), a plant ('tomato care'), prevention, or how to use the app.",
              "इसका उत्तर मेरे पास अभी नहीं है। किसी रोग (जैसे 'late blight'), पौधे ('टमाटर की देखभाल'), बचाव या ऐप के उपयोग के बारे में पूछें।")


def _tab_assistant():
    st.markdown("### " + tr("💬 Plant Assistant", "💬 पौधा सहायक"))
    st.caption(tr(
        "A built-in helper that answers from the app's own plant and disease knowledge. It works offline and is not a live AI chatbot.",
        "यह ऐप की अपनी पौधा और रोग जानकारी से जवाब देने वाला सहायक है। यह ऑफ़लाइन चलता है और लाइव एआई चैटबॉट नहीं है।",
    ))
    if "pro_chat" not in st.session_state:
        st.session_state["pro_chat"] = []

    for msg in st.session_state["pro_chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    samples = [
        tr("Tell me about late blight", "late blight के बारे में बताओ"),
        tr("Tomato care", "टमाटर की देखभाल"),
        tr("How to prevent plant diseases?", "पौधों के रोगों से बचाव कैसे करें?"),
    ]
    clicked = None
    for n, (col, s) in enumerate(zip(st.columns(3), samples)):
        if col.button(s, key=f"pro_sample_{n}", **STRETCH):
            clicked = s

    # A form works inside tabs on every Streamlit version and clears itself.
    with st.form("pro_chat_form", clear_on_submit=True):
        typed = st.text_input(
            tr("Ask about a disease, a plant, or the app...", "रोग, पौधे या ऐप के बारे में पूछें..."),
            key="pro_chat_text",
        )
        sent = st.form_submit_button(tr("Ask", "पूछें"))

    question = clicked or (typed.strip() if sent and typed.strip() else None)
    if question:
        answer = assistant_answer(question)
        st.session_state["pro_chat"].append({"role": "user", "text": question})
        st.session_state["pro_chat"].append({"role": "assistant", "text": answer})
        st.rerun()

    if st.session_state["pro_chat"]:
        if st.button(tr("🧹 Clear chat", "🧹 चैट साफ़ करें"), key="pro_chat_clear"):
            st.session_state["pro_chat"] = []
            st.rerun()


# ---------- Pro sub-tab: Learn & Quiz ----------
def _tab_quiz():
    idx = 1 if LANG == "hi" else 0
    st.markdown("### " + tr("🎓 Learn & Quiz", "🎓 सीखें और क्विज़"))
    tip = TIPS[date.today().toordinal() % len(TIPS)]
    st.success(f"🌤️ **{tr('Tip of the day', 'आज का सुझाव')}:** {tip[idx]}")

    st.markdown("#### " + tr("Plant health quiz", "पौधा स्वास्थ्य क्विज़"))
    with st.form("pro_quiz_form"):
        for n, (question, options, _correct, _why) in enumerate(QUIZ):
            st.radio(
                f"{n + 1}. {question[idx]}",
                options=list(range(len(options))),
                format_func=lambda i, opts=options: opts[i][idx],
                index=None,
                key=f"pro_quiz_q{n}",
            )
        submitted = st.form_submit_button(tr("Check my answers", "मेरे उत्तर जाँचें"))

    if submitted:
        score = 0
        for n, (question, options, correct, why) in enumerate(QUIZ):
            chosen = st.session_state.get(f"pro_quiz_q{n}")
            if chosen == correct:
                score += 1
                st.success(f"{n + 1}. ✅ {why[idx]}")
            else:
                st.error(f"{n + 1}. ❌ {tr('Correct answer', 'सही उत्तर')}: {options[correct][idx]}. {why[idx]}")
        st.markdown(f"### {tr('Your score', 'आपका स्कोर')}: {score}/{len(QUIZ)}")
        if score == len(QUIZ):
            st.balloons()


# ---------- Pro Tools entry point ----------
def _safe(fn):
    """Run one Pro sub-tab; if it fails, show the error instead of crashing the app."""
    try:
        fn()
    except Exception as exc:
        st.error(tr(f"This tool had a problem: {exc}", f"इस टूल में समस्या आई: {exc}"))


def render_pro_tools():
    st.markdown("## " + tr("🚀 Pro Tools", "🚀 प्रो टूल्स"))
    st.write(tr(
        "Extra tools to understand your plants better: colour analysis, AI focus map, before/after comparison, risk advisor, care guide, assistant and quiz.",
        "अपने पौधों को बेहतर समझने के अतिरिक्त टूल: रंग विश्लेषण, एआई फ़ोकस मैप, पहले/बाद की तुलना, जोखिम सलाहकार, देखभाल गाइड, सहायक और क्विज़।",
    ))

    p1, p2, p3, p4, p5, p6 = st.tabs([
        tr("🧠 Insights", "🧠 गहरी जाँच"),
        tr("⚖️ Compare", "⚖️ तुलना"),
        tr("🌦️ Risk", "🌦️ जोखिम"),
        tr("🌿 Care", "🌿 देखभाल"),
        tr("💬 Assistant", "💬 सहायक"),
        tr("🎓 Quiz", "🎓 क्विज़"),
    ])
    with p1:
        _safe(_tab_insights)
    with p2:
        _safe(_tab_compare)
    with p3:
        _safe(_tab_risk)
    with p4:
        _safe(_tab_care)
    with p5:
        _safe(_tab_assistant)
    with p6:
        _safe(_tab_quiz)


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

    .hero-badges span {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #ffffff;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        margin: 0.6rem 0.4rem 0 0;
        font-size: 0.85rem;
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
        border-left: 6px solid #9bc5a2;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 110px;
        box-shadow: 0 4px 14px rgba(30, 80, 45, 0.05);
    }

    .info-card.ok { border-left-color: #2e9d5b; }
    .info-card.warn { border-left-color: #e0a100; }
    .info-card.bad { border-left-color: #d64545; }

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

    .chip {
        display: inline-block;
        background: #e6f4e8;
        color: #174b32;
        border: 1px solid #cfe5d2;
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        margin: 0.25rem 0.3rem 0.25rem 0;
        font-size: 0.92rem;
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


# ---------------- SIDEBAR (language first) ----------------
with st.sidebar:
    language_choice = st.radio(
        "🌐 Language / भाषा", ["English", "हिन्दी"], horizontal=True
    )
    LANG = "hi" if language_choice == "हिन्दी" else "en"

    st.markdown("## 🌿 PlantHealth AI")
    st.caption(t("side_caption"))
    st.divider()

    st.markdown(f"### {t('side_howto')}")
    for key in ("side_1", "side_2", "side_3", "side_4"):
        st.write(t(key))

    st.divider()
    st.markdown(f"### {t('side_tips')}")
    for key in ("side_tip1", "side_tip2", "side_tip3", "side_tip4"):
        st.write(t(key))

    st.divider()
    st.caption(t("side_proto"))


# ---------------- RESULT DISPLAY ----------------
def render_result(res):
    """Show a saved analysis result (kept in session state)."""
    confidence = res["confidence"]
    selected_plant = res["selected_plant"]
    a = assess(res["best_class"], confidence, selected_plant)
    status_text = t(a["status_key"])

    if a["mismatch"]:
        st.error(
            t("mismatch_msg", selected=selected_plant, predicted=a["predicted_plant"]),
            icon="🚨",
        )

    st.markdown(
        f"""
        <div class="info-card {a['style']}">
            <div class="info-label">{t('most_likely')}</div>
            <div class="info-value">{a['display_class']}</div>
            <div class="info-label" style="margin-top:0.6rem;">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    score_col, plant_col = st.columns(2)
    with score_col:
        st.metric(t("model_score"), f"{confidence:.2f}%")
    with plant_col:
        st.metric(
            t("selected_plant"),
            PLANT_HI.get(selected_plant, selected_plant) if LANG == "hi" else selected_plant,
        )
    st.progress(min(max(confidence / 100, 0.0), 1.0))

    st.markdown(f"#### {t('top3')}")
    for prediction in res["top"]:
        label = format_class_name(prediction["disease"])
        score = prediction["confidence"]
        st.write(f"**{label}**")
        st.progress(min(max(score / 100, 0.0), 1.0))
        st.caption(t("score_caption", score=score))

    if a["is_background"]:
        st.warning(t("bg_warn"))
    elif a["low"]:
        st.warning(t("low_warn"))
    elif a["is_healthy"] and not a["mismatch"]:
        st.success(t("healthy_ok"))
    elif not a["mismatch"]:
        st.info(t("disease_info"))

    st.caption(t("conf_note"))

    guidance = build_guidance(res, a)
    st.markdown(f"### {t('care_header')}")
    st.markdown(f"**{guidance['title']}**")
    st.write(guidance["about"])
    st.markdown(f"#### {t('next_steps')}")
    st.markdown("\n".join(f"- {tip}" for tip in guidance["care"]))

    # Downloads
    st.download_button(
        label=t("dl_txt"),
        data=build_text_report(res, a, guidance),
        file_name="plant_health_result.txt",
        mime="text/plain",
        **STRETCH,
    )
    if HAS_PDF:
        try:
            pdf_bytes = build_pdf_report(res, a)
            if pdf_bytes:
                st.download_button(
                    label=t("dl_pdf"),
                    data=pdf_bytes,
                    file_name="plant_health_report.pdf",
                    mime="application/pdf",
                    **STRETCH,
                )
                if LANG == "hi":
                    st.caption(t("pdf_note"))
        except Exception:
            pass

    # Save to My Garden
    st.markdown(f"#### {t('garden_title')}")
    garden_name = st.text_input(t("garden_name"), key="garden_name_input")
    if st.button(t("garden_btn"), key="garden_save_btn", **STRETCH):
        final_name = garden_name.strip() or selected_plant
        try:
            garden_add(
                final_name, selected_plant, a["display_class"],
                confidence, a["good"],
            )
            st.success(t("garden_saved", name=final_name))
        except Exception as exc:
            st.error(t("garden_fail", err=exc))


# ---------------- HERO ----------------
st.markdown(
    f"""
<div class="hero">
    <h1>{t('hero_h1')}</h1>
    <p>{t('hero_p')}</p>
    <div class="hero-badges">
        <span>{t('badge1')}</span>
        <span>{t('badge2')}</span>
        <span>{t('badge3')}</span>
        <span>{t('badge4')}</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.info(t("info_box"))

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

(tab_analyze, tab_stats, tab_history, tab_library,
 tab_batch, tab_garden, tab_pro, tab_about) = st.tabs([
    t("tab_analyze"), t("tab_stats"), t("tab_history"), t("tab_library"),
    t("tab_batch"), t("tab_garden"), t("tab_pro"), t("tab_about"),
])


def load_history():
    try:
        return get_prediction_history()
    except Exception as exc:
        st.error(t("hist_load_err", err=exc))
        return []


def list_samples():
    try:
        if SAMPLES_DIR.is_dir():
            return sorted(
                p for p in SAMPLES_DIR.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
    except Exception:
        pass
    return []


def image_to_jpeg_bytes(image):
    thumb = image.copy()
    thumb.thumbnail((700, 700))
    out = io.BytesIO()
    thumb.save(out, format="JPEG", quality=85)
    return out.getvalue()


# ---------------- TAB 1: ANALYZE ----------------
with tab_analyze:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f'<div class="section-title">{t("sec_add")}</div>', unsafe_allow_html=True)

        selected_plant = st.selectbox(
            t("which_plant"),
            PLANT_OPTIONS,
            index=PLANT_OPTIONS.index("Tomato"),
            format_func=plant_label,
            help=t("which_plant_help"),
            key="plant_main",
        )

        samples = list_samples()
        source_options = [t("src_upload"), t("src_camera")]
        if samples:
            source_options.append(t("src_sample"))
        source = st.radio(t("photo_source"), source_options, horizontal=True)

        photo = None
        sample_path = None
        if source == t("src_upload"):
            photo = st.file_uploader(
                t("choose_file"), type=["jpg", "jpeg", "png"], help=t("choose_help")
            )
        elif source == t("src_camera"):
            photo = st.camera_input(t("take_photo"))
        else:
            sample_path = st.selectbox(
                t("pick_sample"), samples, format_func=lambda p: p.stem
            )

        image = None
        file_name = "camera_photo.jpg"

        try:
            if photo is not None:
                file_name = getattr(photo, "name", None) or "camera_photo.jpg"
                image = Image.open(photo).convert("RGB")
            elif sample_path is not None:
                file_name = sample_path.name
                image = Image.open(sample_path).convert("RGB")
        except Exception:
            st.error(t("cant_open"))
            image = None

        if image is not None:
            st.image(image, caption=t("your_photo"))
            for warning in check_image_quality(image):
                st.warning(warning)

        analyze = st.button(
            t("analyze_btn"), type="primary", disabled=image is None, **STRETCH
        )

    with right:
        st.markdown(f'<div class="section-title">{t("sec_results")}</div>', unsafe_allow_html=True)

        if analyze and image is not None:
            try:
                with st.spinner(t("analyzing")):
                    best_class, confidence, top_predictions = predict_plant_disease(image)

                # Save the prediction permanently in SQLite.
                save_prediction(file_name, best_class, confidence)

                st.session_state["last_result"] = {
                    "file_name": file_name,
                    "selected_plant": selected_plant,
                    "best_class": best_class,
                    "confidence": float(confidence),
                    "top": list(top_predictions),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_bytes": image_to_jpeg_bytes(image),
                }
            except Exception as exc:
                st.error(t("analysis_err", err=exc))

        result = st.session_state["last_result"]

        if result is not None:
            try:
                render_result(result)
            except Exception as exc:
                st.error(t("display_err", err=exc))

            if st.button(t("clear_result"), key="clear_result_btn", **STRETCH):
                st.session_state["last_result"] = None
                st.rerun()
        else:
            st.info(t("prompt_result"))


# ---------------- TAB 2: ANALYTICS ----------------
with tab_stats:
    st.markdown(f"## {t('stats_title')}")
    history = load_history()

    try:
        if history:
            total_predictions = len(history)
            average_confidence = (
                sum(record["Confidence"] for record in history) / total_predictions
            )

            disease_counts = {}
            healthy_count = 0

            for record in history:
                prediction_name = format_class_name(record["Prediction"])
                disease_counts[prediction_name] = disease_counts.get(prediction_name, 0) + 1
                if "healthy" in prediction_name.lower():
                    healthy_count += 1

            most_common_prediction = max(disease_counts, key=disease_counts.get)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(t("m_total"), total_predictions)
            with col2:
                st.metric(t("m_avg"), f"{average_confidence:.2f}%")
            with col3:
                st.metric(t("m_healthy"), healthy_count)
            with col4:
                st.metric(t("m_common"), most_common_prediction, help=t("m_common_help"))

            st.markdown(f"### {t('top5')}")

            top_five = sorted(
                disease_counts.items(), key=lambda item: item[1], reverse=True
            )[:5]

            chart_data = {
                t("col_class"): [item[0] for item in top_five],
                t("col_count"): [item[1] for item in top_five],
            }

            try:
                st.bar_chart(chart_data, x=t("col_class"), y=t("col_count"), horizontal=True)
            except TypeError:
                # Older Streamlit versions do not support horizontal charts.
                st.bar_chart(chart_data, x=t("col_class"), y=t("col_count"))

            st.caption(t("stats_note"))
        else:
            st.info(t("stats_empty"))
    except Exception as exc:
        st.error(t("stats_err", err=exc))


# ---------------- TAB 3: HISTORY ----------------
with tab_history:
    st.markdown(f"## {t('hist_title')}")
    st.caption(t("hist_caption"))
    history = load_history()

    if st.session_state.pop("hist_cleared_flag", False):
        st.success(t("hist_cleared"))

    try:
        if history:
            search = st.text_input(
                t("hist_search"), placeholder=t("hist_search_ph"), key="hist_search_box"
            ).strip().lower()

            display_history = []
            for record in history:
                row = {
                    t("col_time"): record["Time"],
                    t("col_image"): record["Image"],
                    t("col_pred"): format_class_name(record["Prediction"]),
                    t("col_conf"): f'{record["Confidence"]:.2f}%',
                }
                haystack = f'{record["Image"]} {format_class_name(record["Prediction"])}'.lower()
                if not search or search in haystack:
                    display_history.append(row)

            if display_history:
                st.dataframe(display_history, hide_index=True)
                st.caption(t("hist_showing", n=len(display_history), total=len(history)))
            else:
                st.info(t("hist_nomatch"))

            confirm = st.checkbox(t("hist_confirm"), key="hist_confirm_box")
            if st.button(t("hist_clear"), disabled=not confirm, key="hist_clear_btn"):
                clear_prediction_history()
                st.session_state["hist_cleared_flag"] = True
                st.rerun()
        else:
            st.info(t("hist_empty"))
    except Exception as exc:
        st.error(t("hist_err", err=exc))

    # Download the full history as CSV.
    if history:
        try:
            csv_buffer = io.StringIO()
            fieldnames = ["Prediction Time", "Image Name", "Predicted Disease", "Confidence (%)"]
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
                label=t("hist_dl"),
                data=csv_buffer.getvalue(),
                file_name="plant_health_prediction_history.csv",
                mime="text/csv",
                **STRETCH,
            )
        except Exception as exc:
            st.error(t("csv_err", err=exc))


# ---------------- TAB 4: DISEASE LIBRARY ----------------
with tab_library:
    st.markdown(f"## {t('lib_title')}")
    st.write(t("lib_intro"))

    lib_col1, lib_col2 = st.columns([2, 1])
    with lib_col1:
        lib_search = st.text_input(
            t("lib_search"), placeholder=t("lib_search_ph"), key="lib_search_box"
        ).strip().lower()
    with lib_col2:
        lib_plant = st.selectbox(
            t("lib_filter"),
            ["__all__"] + PLANT_OPTIONS,
            format_func=lambda p: t("lib_all") if p == "__all__" else plant_label(p),
            key="lib_plant_box",
        )

    lib_idx = 1 if LANG == "hi" else 0
    lib_shown = 0
    for item in DISEASES:
        if item["name"] is None:
            continue  # duplicate keyword entry, listed once above
        if lib_plant != "__all__" and lib_plant not in item["plants"]:
            continue
        plant_text = " ".join(
            f"{p} {PLANT_HI.get(p, '')}" for p in item["plants"]
        )
        haystack = f'{item["name"][0]} {item["name"][1]} {plant_text}'.lower()
        if lib_search and lib_search not in haystack:
            continue

        lib_shown += 1
        with st.expander(f"🍂 {item['name'][lib_idx]}"):
            plants_label = ", ".join(plant_label(p) for p in item["plants"])
            st.markdown(f"**{t('lib_affects')}:** {plants_label}")
            st.markdown(f"**{t('lib_symptoms')}:** {item['about'][lib_idx]}")

    if lib_shown == 0:
        st.info(t("lib_none"))

    st.markdown(f"### {t('lib_prevent_title')}")
    st.markdown("\n".join(f"- {tip}" for tip in t("lib_prevent")))
    st.caption(t("conf_note"))


# ---------------- TAB 5: BATCH ----------------
with tab_batch:
    st.markdown(f"## {t('batch_title')}")
    st.write(t("batch_intro", n=MAX_BATCH))

    batch_plant = st.selectbox(
        t("batch_plant"),
        PLANT_OPTIONS,
        index=PLANT_OPTIONS.index("Tomato"),
        format_func=plant_label,
        key="plant_batch",
    )
    batch_files = st.file_uploader(
        t("batch_upload", n=MAX_BATCH),
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files and len(batch_files) > MAX_BATCH:
        st.warning(t("batch_limit", n=MAX_BATCH))
        batch_files = batch_files[:MAX_BATCH]

    if st.button(t("batch_btn"), type="primary", disabled=not batch_files, key="batch_btn", **STRETCH):
        rows = []
        total = len(batch_files)
        progress = st.progress(0.0)
        status_box = st.empty()
        for i, up in enumerate(batch_files, start=1):
            name = getattr(up, "name", f"image_{i}")
            try:
                status_box.caption(t("batch_progress", i=i, n=total))
                img = Image.open(up).convert("RGB")
                best_class, confidence, _top = predict_plant_disease(img)
                save_prediction(name, best_class, confidence)
                a = assess(best_class, float(confidence), batch_plant)
                rows.append({
                    "name": name,
                    "image_bytes": image_to_jpeg_bytes(img),
                    "class": a["display_class"],
                    "score": float(confidence),
                    "status_key": a["status_key"],
                })
            except Exception as exc:
                st.error(t("batch_fail", name=name, err=exc))
            progress.progress(i / total)
        status_box.empty()
        st.session_state["batch_results"] = rows

    batch_rows = st.session_state["batch_results"]
    if batch_rows:
        st.markdown(f"### {t('batch_summary')}")
        thumb_cols = st.columns(min(len(batch_rows), MAX_BATCH))
        for col, row in zip(thumb_cols, batch_rows):
            with col:
                st.image(row["image_bytes"], caption=row["name"])

        table_rows = [
            {
                t("col_image"): row["name"],
                t("col_pred"): row["class"],
                t("col_score"): f'{row["score"]:.2f}%',
                t("col_status"): t(row["status_key"]),
            }
            for row in batch_rows
        ]
        st.dataframe(table_rows, hide_index=True)

        try:
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=["Image", "Prediction", "Score (%)", "Status"])
            writer.writeheader()
            for row in batch_rows:
                writer.writerow({
                    "Image": row["name"],
                    "Prediction": row["class"],
                    "Score (%)": f'{row["score"]:.2f}',
                    "Status": t(row["status_key"], "en"),
                })
            st.download_button(
                t("batch_dl"), data=out.getvalue(),
                file_name="plant_health_batch_results.csv", mime="text/csv", **STRETCH,
            )
        except Exception as exc:
            st.error(t("csv_err", err=exc))


# ---------------- TAB 6: MY GARDEN ----------------
with tab_garden:
    st.markdown(f"## {t('garden_page_title')}")
    st.write(t("garden_intro"))

    deleted_name = st.session_state.pop("garden_deleted_flag", None)
    if deleted_name:
        st.success(t("garden_deleted", name=deleted_name))

    try:
        names = garden_names()
    except Exception as exc:
        names = []
        st.error(t("garden_fail", err=exc))

    if not names:
        st.info(t("garden_empty"))
    else:
        chosen = st.selectbox(t("garden_pick"), names, key="garden_pick_box")
        records = garden_records(chosen)

        if records:
            last = records[-1]
            m1, m2 = st.columns(2)
            with m1:
                st.metric(t("garden_checkins"), len(records))
            with m2:
                st.metric(t("garden_last"), t("garden_ok") if last[4] else t("garden_bad"))

            st.markdown(f"#### {t('garden_chart')}")
            chart_df = pd.DataFrame({
                t("garden_col_date"): [r[0] for r in records],
                t("garden_col_health"): [int(r[4]) for r in records],
            }).set_index(t("garden_col_date"))
            try:
                st.line_chart(chart_df)
            except Exception:
                st.dataframe(chart_df)

            st.dataframe(
                [
                    {
                        t("garden_col_date"): r[0],
                        t("col_pred"): r[2],
                        t("col_conf"): f"{r[3]:.2f}%",
                        t("garden_col_status"): t("garden_ok") if r[4] else t("garden_bad"),
                    }
                    for r in records
                ],
                hide_index=True,
            )

        sure = st.checkbox(t("garden_confirm"), key="garden_confirm_box")
        if st.button(t("garden_delete"), disabled=not sure, key="garden_delete_btn"):
            try:
                garden_delete(chosen)
                st.session_state["garden_deleted_flag"] = chosen
                st.rerun()
            except Exception as exc:
                st.error(t("garden_fail", err=exc))

    st.caption(t("garden_note"))


# ---------------- TAB 7: PRO TOOLS ----------------
with tab_pro:
    render_pro_tools()


# ---------------- TAB 8: ABOUT ----------------
with tab_about:
    st.markdown(f"## {t('about_title')}")
    st.write(t("about_intro"))

    st.markdown(f"### {t('about_plants')}")
    chips = "".join(
        f'<span class="chip">{plant_label(p)}</span>' for p in PLANT_OPTIONS
    )
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown(f"### {t('about_how')}")
    st.markdown(t("about_steps"))

    st.markdown(f"### {t('about_good')}")
    st.markdown(t("about_notes"))


# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    f'<p class="small-note" style="text-align:center;">{t("footer")}</p>',
    unsafe_allow_html=True,
)