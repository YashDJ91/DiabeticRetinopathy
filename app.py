import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(
    page_title="RetinaScan Pro",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS Styling ----

def local_css():
    st.markdown("""
    <style>
    :root {
        --primary: #3498db;
        --primary-dark: #2980b9;
        --secondary: #2c3e50;
        --accent: #e74c3c;
        --success: #2ecc71;
        --warning: #f39c12;
        --danger: #e74c3c;
        --light: #f8f9fa;
        --dark: #34495e;
        --white: #ffffff;
    }
    
    
    
   .main {
       background-image: url("main/bj.jpg");
       border-radius: 15px;
       box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
       padding: 2rem;
       margin-bottom: 2rem;
       backdrop-filter: blur(5px);
}
    
    .title {
        color: var(--secondary);
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: var(--dark);
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.85);
        padding: 1.5rem;
        border-radius: 0 15px 15px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        backdrop-filter: blur(5px);
    }
    
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        background-color: rgba(255,255,255,0.7);
    }
    
    .nav-item:hover {
        background-color: rgba(52, 152, 219, 0.2);
        color: var(--primary);
    }
    
    .nav-item.active {
        background-color: var(--primary);
        color: white !important;
    }
    
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-left: 5px solid;
        background-color: rgba(255,255,255,0.8);
    }
    
    .severity-0 { 
        border-left-color: var(--success) !important; 
        background-color: rgba(46, 204, 113, 0.1);
    }
    .severity-1 { 
        border-left-color: var(--warning) !important; 
        background-color: rgba(243, 156, 18, 0.1);
    }
    .severity-2 { 
        border-left-color: #e67e22 !important; 
        background-color: rgba(230, 126, 34, 0.1);
    }
    .severity-3 { 
        border-left-color: #d35400 !important; 
        background-color: rgba(211, 84, 0, 0.1);
    }
    .severity-4 { 
        border-left-color: var(--danger) !important; 
        background-color: rgba(231, 76, 60, 0.1);
    }
    
    .upload-area {
        border: 2px dashed var(--primary);
        border-radius: 12px;
        padding: 3rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    .upload-area:hover {
        background-color: rgba(52, 152, 219, 0.15);
    }
    
    .info-box {
        background-color: rgba(255,255,255,0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    .stage-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
    }
    
    @media (max-width: 768px) {
        .title {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ---- Load Model ----
@st.cache_resource
def load_trained_model():
    model = load_model("diabetic_retinopathy_model.h5")
    return model

model = load_trained_model()

# ---- Class Names and Descriptions ----
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
class_descriptions = [
    "No signs of diabetic retinopathy detected. Regular eye checkups are still recommended for diabetics.",
    "Mild nonproliferative retinopathy with microaneurysms only. Early stage that may not require treatment but needs monitoring.",
    "Moderate nonproliferative retinopathy with more extensive vascular changes. May need treatment depending on symptoms.",
    "Severe nonproliferative retinopathy with extensive vascular damage. Usually requires prompt treatment.",
    "Proliferative diabetic retinopathy, the most advanced stage with new blood vessel growth. Requires immediate treatment."
]
class_colors = ["#2ecc71", "#f39c12", "#e67e22", "#d35400", "#e74c3c"]

# ---- Navigation ----
def navigation():
    st.sidebar.title("🔍 Navigation")
    menu = ["🏠 Home", "ℹ️ About Us", "📞 Contact Us"]
    choice = st.sidebar.radio("", menu, index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">ℹ️ About Diabetic Retinopathy</h3>
        <p>Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).</p>
        <p><strong>Early detection can prevent vision loss.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    return choice

current_page = navigation()

# ---- Home Page ----
if current_page == "🏠 Home":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="title">RetinaScan Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Advanced AI-powered Diabetic Retinopathy Detection System</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">📌 How It Works</h3>
            <ol style="padding-left: 1.5rem;">
                <li>Upload a retina fundus image (JPG/PNG)</li>
                <li>Our AI analyzes the image in seconds</li>
                <li>Get instant results with severity assessment</li>
                <li>Receive recommendations based on findings</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://wiseyak.com/wp-content/uploads/2025/03/eye-scan-7412898-scaled.jpg", 
                caption="Advanced Retina Scanning Technology", use_container_width=True)
    
    st.markdown("---")
    
    # Upload section
    st.markdown("## 📁 Upload Retina Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="file_uploader")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Retina Image", use_container_width=True)
        
        if st.button("🔍 Analyze Image", key="analyze_btn"):
            with st.spinner("🧠 AI is analyzing your retina image..."):
                # Ensure consistent RGB format
                image_rgb = image.convert("RGB")
                
                # Resize to match model's expected input
                img_resized = image_rgb.resize((224, 224))
                
                # Convert to array and normalize
                img_array = img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction = model.predict(img_array)
                predicted_class = int(np.argmax(prediction, axis=1)[0])
            
            st.markdown("## 📊 Analysis Results")
            
            # Result card
            st.markdown(f"""
            <div class="result-card severity-{predicted_class}">
                <div style="display: flex; align-items: center;">
                    <span class="stage-indicator" style="background-color: {class_colors[predicted_class]};"></span>
                    <h3 style="margin: 0; color: {class_colors[predicted_class]};">{class_names[predicted_class]}</h3>
                </div>
                <p style="margin: 0.5rem 0 0 30px;">{class_descriptions[predicted_class]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations based on severity
            st.markdown("## 📝 Recommendations")
            if predicted_class == 0:
                st.success("""
                ✅ **No action required** for diabetic retinopathy.  
                However, if you have diabetes, regular annual eye exams are still recommended.
                """)
            elif predicted_class == 1:
                st.warning("""
                ⚠️ **Schedule a follow-up** with an ophthalmologist within 6-12 months.  
                Maintain good blood sugar control to prevent progression.
                """)
            elif predicted_class == 2:
                st.warning("""
                ⚠️ **Consult an eye specialist** within 3-6 months.  
                You may need additional testing like fluorescein angiography.
                """)
            elif predicted_class == 3:
                st.error("""
                ❗ **Urgent consultation needed** with a retina specialist within 1 month.  
                Treatment options may include laser therapy or injections.
                """)
            else:
                st.error("""
                🚨 **Immediate medical attention required**.  
                Contact a retina specialist immediately. Proliferative DR can lead to serious vision loss without prompt treatment.
                """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---- About Us Page ----
elif current_page == "ℹ️ About Us":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.markdown('<div class="title">About RetinaScan Pro</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem;">Revolutionizing diabetic retinopathy detection through artificial intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## Our Mission
    At RetinaScan Pro, we're committed to early detection of diabetic retinopathy to prevent vision loss. 
    Our AI-powered platform provides accurate, instant assessments of retina images, helping both patients 
    and healthcare providers identify potential issues before they become serious.
    
    ## Technology
    Our system uses a state-of-the-art deep learning model trained on thousands of retina images 
    from diverse populations. The model achieves medical-grade accuracy in classifying diabetic 
    retinopathy into five severity stages according to international standards.
    """)
    
    st.markdown("---")
    
    # Feature highlights without images
    st.markdown("## ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AI-Powered Detection")
        st.markdown("Our deep learning model provides accurate classification of diabetic retinopathy stages with medical-grade precision.")
    
    with col2:
        st.markdown("### Rapid Analysis")
        st.markdown("Get results in seconds, enabling timely interventions when they matter most.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Contact Us Page ----
elif current_page == "📞 Contact Us":
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.markdown('<div class="title">Contact Us</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Have questions or feedback? We\'d love to hear from you!</div>', unsafe_allow_html=True)
    
    with st.form("contact_form"):
        st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">✉️ Send Us a Message</h3>
        """, unsafe_allow_html=True)
        
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        subject = st.selectbox("Subject", ["General Inquiry", "Technical Support", "Partnership", "Feedback"])
        message = st.text_area("Your Message", height=150)
        
        submit_button = st.form_submit_button("Send Message")
        
        if submit_button:
            if name and email and message:
                st.success("Thank you for your message! We'll get back to you within 24-48 hours.")
            else:
                st.warning("Please fill in all required fields.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)