import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests
import google.generativeai as genai

# ==== GEMINI CONFIGURATION ====

GOOGLE_API_KEY = "AIzaSyBZqGn9XXw8ML1uUHaqjulYOGwyHhfa2as"
genai.configure(api_key=GOOGLE_API_KEY)
chat_model = genai.GenerativeModel(
    model_name="models/gemini-2.0-flash",
    system_instruction="You are a helpful PCOS assistant. Provide empathetic, accurate information about Polycystic Ovary Syndrome (PCOS), its symptoms, treatments, and management strategies. Do not provide medical diagnosis."
)
chat_session = chat_model.start_chat()  # Remove system message from history


# ==== MODEL CONFIGURATION ====
MODEL_URL = "https://github.com/Taneesha3105/PCOS_detection/releases/download/v1.0.0/PCOS_resnet18_model.pth"
MODEL_PATH = "PCOS_resnet18_model.pth"
CLASS_NAMES = ['PCOS', 'No PCOS']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== PAGE CONFIGURATION ====
st.set_page_config(
    page_title="PCOSense Companion", 
    layout="wide", 
    page_icon="🌸",
    initial_sidebar_state="expanded"
)

# ==== CUSTOM CSS ====
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    * {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #f8f1f4;
    }
    .stApp {
        background-image: linear-gradient(135deg, #ffffff 0%, #fff9fb 100%);
    }
    .big-font {
        font-size: 42px !important;
        font-weight: 700;
        color: #d94c63;
        text-align: center;
        margin: 20px 0;
    }
    .medium-font {
        font-size: 22px !important;
        color: #444;
        text-align: center;
    }
    .small-font {
        font-size: 16px !important;
        color: #666;
    }
    .stButton>button {
        background-color: #d94c63;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #c03651;
        transform: scale(1.05);
    }
    .info-box {
        background-color: #fff;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #d94c63;
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
    }
    .pcos-positive {
        background-color: rgba(217, 76, 99, 0.1);
        border: 2px solid #d94c63;
    }
    .pcos-negative {
        background-color: rgba(75, 192, 192, 0.1);
        border: 2px solid #4bc0c0;
    }
    .tabs-font {
        font-size: 18px !important;
        font-weight: bold;
    }
    div[data-testid="stFileUploader"] > label > div {
        display: none;
    }
    .banner-image {
        border-radius: 20px;
        margin: 25px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==== SIDEBAR ====
with st.sidebar:
    st.markdown('<div class="medium-font">PCOSense Companion🚺</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="small-font">A woman\'s best friend for PCOS detection and support❤</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Quick Facts")
    st.info("• PCOS affects about 6% Indian women")
    st.info("• It affects 5% to 10% of women in their reproductive age")
    st.info("• PCOS is a leading cause of female infertility")
    st.info("• Early diagnosis can help manage symptoms effectively")
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("📚[PCOS Diet Guide](https://www.hopkinsmedicine.org/health/wellness-and-prevention/pcos-diet)")
    st.markdown("🧘‍♀[Exercise Recommendations](https://www.healthline.com/health/womens-health/exercise-for-pcos)")
    st.markdown("👩‍⚕[Find a Specialist](https://www.pennmedicine.org/for-patients-and-visitors/find-a-program-or-service/penn-fertility-care/pcos-program)")

# ==== MAIN PAGE ====
st.markdown('<div class="big-font">🌸PCOSense Companion🌸</div>', unsafe_allow_html=True)
st.markdown('<div class="medium-font">AI-powered PCOS detection and support system</div>', unsafe_allow_html=True)

# ==== BANNER IMAGE ====
banner_path = "PCOS.png"
if os.path.exists(banner_path):
    st.image(
        banner_path,
        use_container_width=True,
        caption="Empowering Women's Health Through AI",
        output_format="PNG"
    )

# ==== TABS ====
tab1, tab2, tab3 = st.tabs(["🔍PCOS Detection", "❓About PCOS", "💬Ask An Expert"])

with tab1:
    st.markdown('<div class="small-font">Upload an ultrasound image to detect signs of Polycystic Ovary Syndrome (PCOS)</div>', unsafe_allow_html=True)
    
    # ==== DOWNLOAD MODEL ====
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

    @st.cache_resource
    def load_model():
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model

    model = load_model()

    # ==== TRANSFORMS ====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # ==== IMAGE UPLOAD ====
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            with col1:
                st.image(image, caption="📷Uploaded Ultrasound Image", use_container_width=True)

            with col2:
                with st.spinner("🔍Analyzing image..."):
                    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
                        prediction = CLASS_NAMES[predicted.item()]
                
                if prediction == "PCOS":
                    st.markdown(f"""
                    <div class="prediction-box pcos-positive">
                        <h2>🔍Result: PCOS Detected</h2>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>This ultrasound image shows potential signs of Polycystic Ovary Syndrome.</p>
                        <p><b>Important:</b> This is not a medical diagnosis. Please consult a healthcare professional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box pcos-negative">
                        <h2>🔍Result: No PCOS Detected</h2>
                        <p>Confidence: {confidence * 100:.2f}%</p>
                        <p>This ultrasound image does not show typical signs of Polycystic Ovary Syndrome.</p>
                        <p><b>Note:</b> Always consult with healthcare professionals for proper diagnosis.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error("⚠Invalid image file. Please try again.")
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>How to use the PCOS Detection tool:</h3>
            <ol>
                <li>Upload a clear ultrasound image of the ovaries</li>
                <li>Our AI model will analyze the image</li>
                <li>Results will appear showing whether PCOS indicators are detected</li>
                <li>Remember that this tool is for educational purposes only</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="medium-font">Understanding Polycystic Ovary Syndrome</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>What is PCOS?</h3>
        <p>Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder affecting women of reproductive age. It is marked by irregular or absent periods, higher levels of male hormones (androgens) and multiple small cysts on the ovaries. Common symptoms include irregular menstrual cycles, excess facial or body hair, acne, weight gain and difficulties with fertility. The exact cause is unknown, but factors like genetics, insulin resistance and hormonal imbalance play a role. PCOS can increase the risk of type 2 diabetes, high blood pressure, heart disease and mood disorders. However, early diagnosis and lifestyle changes, such as healthy eating and exercise, can help manage symptoms and reduce long-term health risks.</p>
    </div>
    """, unsafe_allow_html=True)

    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>Management Strategies</h3>
            <ul>
                <li>Healthy diet and regular exercise</li>
                <li>Medications to regulate periods</li>
                <li>Treatments for specific symptoms</li>
                <li>Regular monitoring by healthcare professionals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Risk Factors</h3>
            <ul>
                <li>Family history of PCOS</li>
                <li>Obesity</li>
                <li>Insulin resistance</li>
                <li>Sedentary lifestyle</li>
                <li>Hormonal imbalances</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="medium-font">🤖Ask our AI assistant about PCOS</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-font">Get answers to your questions about PCOS symptoms, management and more</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about PCOS..."):
        if not prompt.strip():
            st.error("Please enter a valid question!")
            st.stop()
            
        try:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            with st.spinner("Thinking..."):
                response = chat_session.send_message(prompt)
                response_text = response.text
                
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Redraw chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            st.session_state.messages.pop()  # Remove failed user message



# ==== FOOTER ====
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col1:
    st.markdown("### Connect With Us")
    st.markdown("📱[Instagram](https://instagram.com)")
    st.markdown("🐦[Twitter](https://twitter.com)")
    st.markdown("📘[Facebook](https://facebook.com)")

with footer_col2:
    st.markdown("### Disclaimer")
    st.markdown("""
    <div class="small-font">
    This application provides educational information only. 
    It is not a substitute for professional medical advice. 
    Always consult qualified healthcare providers for medical concerns.
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("### Support")
    st.markdown("❤[Donate](https://example.com)")
    st.markdown("🤝[Volunteer](https://example.com)")
    st.markdown("📧contact@pcosense.org")
