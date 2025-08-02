import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from tensorflow.keras.models import load_model
from fpdf import FPDF
import pickle

st.set_page_config(page_title="Tool Wear Prediction", page_icon="🛠", layout="wide")

# 💡 Background styling with image and improved text contrast
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1605902711622-cfb43c4437b1');
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, .markdown-text-container, .stText, .stMarkdown {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px #000000;
    }
    .highlight {
        font-weight: bold;
        color: #ffeb3b;
        text-shadow: 1px 1px 2px #000;
    }
    .highlight-success {
        font-weight: bold;
        color: #00e676;
    }
    .highlight-warning {
        font-weight: bold;
        color: #ff9100;
    }
    .highlight-danger {
        font-weight: bold;
        color: #ff5252;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #4caf50, #6a1b9a);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔄 Load model and scaler
@st.cache_resource

def load_models():
    tool_model = load_model("model.h5")
    scaler = joblib.load("scaler.joblib")
    return tool_model, scaler

tool_model, scaler = load_models()

with open("scaler_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

top_feature_cols = [
    'No', 'Y1_OutputCurrent', 'clamp_pressure', 'M1_CURRENT_FEEDRATE',
    'X1_OutputCurrent', 'feedrate', 'X1_CommandPosition',
    'X1_ActualPosition', 'Y1_CommandPosition', 'X1_OutputVoltage',
    'Y1_OutputVoltage', 'Z1_CommandPosition', 'S1_OutputCurrent',
    'X1_CurrentFeedback', 'M1_CURRENT_PROGRAM_NUMBER', 'Y1_ActualVelocity',
    'Y1_CommandVelocity', 'S1_ActualAcceleration'
]

user_features = [f for f in top_feature_cols if f != 'No']
sequence_length = 10

feature_guidance = {
    'clamp_pressure': {'Worn': '3–4', 'Unworn': '7–9'},
    'feedrate': {'Worn': '3–10', 'Unworn': '90–140'},
    'M1_CURRENT_FEEDRATE': {'Worn': '3–20', 'Unworn': '90–140'},
    'Y1_OutputCurrent': {'Worn': '320–330', 'Unworn': '90–150'},
}

# 🔧 Navigation
menu = ["Home", "Predict"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("🔨 CNC Tool Wear Prediction")
    st.markdown("""
    <h3 style='color: green;'>About this App 🛠</h3>
    <p>This app predicts <b>Tool Condition (Worn/Unworn)</b>, <b>Machining Finalized</b>, and <b>Visual Inspection</b> status based on CNC sensor data.</p>
    <ul>
        <li>✅ Reduce downtime</li>
        <li>💰 Save costs on tool replacements</li>
        <li>🎯 Improve machining quality</li>
    </ul>
    📌 Powered by Deep Learning (1D CNN) for sequential sensor data analysis.
    """, unsafe_allow_html=True)

elif choice == "Predict":
    st.title("🎛 Tool Wear Prediction Panel ⚙️")
    st.sidebar.header("📊 Feature Input & Guidance")

    with st.sidebar.expander("ℹ️ Feature Ranges"):
        for feat, vals in feature_guidance.items():
            st.write(f"**{feat}** — Worn: {vals['Worn']} | Unworn: {vals['Unworn']}")

    uploaded_file = st.sidebar.file_uploader("📎 Upload CNC CSV File", type=["csv"])
    input_df = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'No' not in df.columns:
                df['No'] = 0.0
            missing = [col for col in top_feature_cols if col not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                input_df = df[top_feature_cols].iloc[:sequence_length]
                if len(input_df) < sequence_length:
                    pad = pd.DataFrame([input_df.iloc[-1]] * (sequence_length - len(input_df)))
                    input_df = pd.concat([input_df, pad], ignore_index=True)
        except Exception as e:
            st.error(f"❗ Error: {e}")

    else:
        st.sidebar.markdown("👈 Or enter values manually:")
        data = {f: st.sidebar.number_input(f, value=0.0) for f in user_features}
        data['No'] = 0.0
        input_df = pd.DataFrame([data] * sequence_length)
    #------------------------------------
    if st.sidebar.button("🚀 Predict") and input_df is not None:
        if (input_df['clamp_pressure'] < 6).any():
            st.warning("⚠️ Warning: Clamping Pressure too low! Risk of faulty parts.")

        # 🧠 Reindex input_df to expected columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0  # Add missing columns with default value

        # Drop any extra columns not in expected
        input_df = input_df[expected_columns]

        # Scale input
        scaled = scaler.transform(input_df)

        if scaled.shape[0] == 0:
            st.error("⚠️ Error: No input data provided for prediction.")
            st.stop()

        # Expected shape for model
        reshaped = np.reshape(scaled, (1, scaled.shape[0], scaled.shape[1]))
        st.write("✅ Reshaped input shape:", reshaped.shape)

        # Prediction
        tool_prob = tool_model.predict(reshaped, verbose=0)[0][0]
        tool_class = int(tool_prob > 0.4)
        tool_label = 'Worn' if tool_class else 'Unworn'
        confidence = tool_prob * 100 if tool_class else (1 - tool_prob) * 100
        
        # 🔄 Dynamic conditions based on experiment data
        if input_df['clamp_pressure'].mean() > 6 and input_df['feedrate'].mean() > 80:
            machining = "Yes"
        else:
            machining = "No"
        visual = "Failed"

        st.subheader("📝 Prediction Results")
        st.success(f"🔧 Tool Condition: {tool_label} ({confidence:.2f}%)")
        st.info(f"🏭 Machining Finalized: {machining}")
        st.warning(f"🔍 Visual Inspection: {visual}")

        report = {
            "Tool Condition": tool_label,
            "Confidence": f"{confidence:.2f}%",
            "Machining Finalized": machining,
            "Visual Inspection": visual
        }
        st.download_button("⬇️ Download CSV", pd.DataFrame([report]).to_csv(index=False), file_name="report.csv")

        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_fill_color(13, 78, 216)
        pdf.set_text_color(255, 255, 255)            
        pdf.cell(0, 12, " CNC Tool Wear Prediction Report ", ln=True, align='C', fill=True)
        pdf.ln(10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Tool Condition: {tool_label} ({confidence:.2f}%)", ln=True)
        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button("⬇️ Download PDF Report", f.read(), file_name="tool_wear_report.pdf")

    #---------------------------
    