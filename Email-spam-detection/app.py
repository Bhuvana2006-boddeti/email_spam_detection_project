import streamlit as st
import pickle
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Email Spam Detector", layout="centered")

# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------- ANIMATED BACKGROUND ----------
def set_bg():
    with open("bg.jpg", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    page_bg = f"""
    <style>

    /* Animated Background */
    .stApp {{
        background: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        animation: zoomMove 30s infinite alternate ease-in-out;
    }}

    @keyframes zoomMove {{
        0% {{ background-size: 100%; }}
        100% {{ background-size: 115%; }}
    }}

    /* Glass Container */
    .block-container {{
        background: rgba(0, 0, 0, 0.65);
        padding: 2.5rem;
        border-radius: 25px;
        backdrop-filter: blur(20px);
        box-shadow: 0 0 40px rgba(0,255,255,0.4);
        border: 1px solid rgba(0,255,255,0.3);

        margin-top: 22vh;   /* pushes content below face */
        animation: fadeIn 1.5s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Title Glow */
    h1 {{
        text-align: center;
        color: white !important;
        animation: glow 2s infinite alternate;
    }}

    @keyframes glow {{
        from {{ text-shadow: 0 0 10px cyan; }}
        to {{ text-shadow: 0 0 30px #00ffff; }}
    }}

    /* Subtitle */
    p {{
        text-align: center;
        color: #ddd;
        font-size: 18px;
    }}

    /* Text Area */
    textarea {{
        border-radius: 15px !important;
        background: rgba(255,255,255,0.95) !important;
        padding: 12px !important;
        transition: 0.3s;
    }}

    textarea:focus {{
        box-shadow: 0 0 20px cyan !important;
    }}

    /* Button Animation */
    div.stButton > button {{
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 12px;
        padding: 10px 25px;
        font-weight: bold;
        transition: 0.4s;
        border: none;
    }}

    div.stButton > button:hover {{
        transform: scale(1.1);
        box-shadow: 0 0 25px #ff4b2b;
    }}

    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)

set_bg()

# ---------- UI ----------
st.markdown("<h1>📧 Email Spam Detection</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter a message to check whether it is Spam or Not Spam</p>", unsafe_allow_html=True)

input_msg = st.text_area("")

if st.button("🔍 Check Message"):
    if input_msg.strip() != "":
        message_vector = vectorizer.transform([input_msg])
        prediction = model.predict(message_vector)
        probability = model.predict_proba(message_vector)

        if prediction[0] == 1:
            st.error(f"🚨 Spam Detected! (Confidence: {probability[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {probability[0][0]*100:.2f}%)")
    else:
        st.warning("Please enter a message.")