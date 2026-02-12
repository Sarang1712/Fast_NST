import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from neural_style import TransformerNet
import io

# --- 1. Page Configuration (Must be first) ---
st.set_page_config(
    page_title="Neural Art Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for "Glassmorphism" Look ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: rgb(20,20,30);
        background: linear-gradient(135deg, rgba(20,20,30,1) 0%, rgba(40,40,60,1) 100%);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card/Container Styling */
    div.css-1r6slb0, div.stButton {
        border-radius: 15px;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ffffff;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Success/Spinner Text */
    .stAlert {
        background-color: rgba(0, 255, 128, 0.1);
        border: 1px solid rgba(0, 255, 128, 0.2);
        color: #00ff80;
    }
    
    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #ff4b1f 0%, #ff9068 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 75, 31, 0.4);
    }
    
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Image Borders */
    img {
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Configuration & Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EDIT THIS: Match the filenames exactly to what you have in your 'models' folder
STYLES = {
    "🎨 Picasso (Cubism)": "models/picasso.pth", 
    "🌌 Starry Night (Van Gogh)": "models/vangogh.pth",
    "🌊 Great Wave (Hokusai)": "models/greatwave.pth",
    "✨ Mosaic (Byzantine)": "models/mosaic.pth"
}

@st.cache_resource
def load_model(model_path):
    try:
        model = TransformerNet()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        return None

def stylize(image, model):
    # Preprocess
    content_transform = transforms.Compose([
        transforms.Resize((640, 640)), # Higher res for better visuals
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(content_image)

    # Postprocess
    img = output[0].clone().clamp(0, 255).cpu().detach().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(img)

# --- 4. Main App Layout ---

# Sidebar Controls
with st.sidebar:
    st.title("🎛️ Studio Controls")
    st.write("Configurate your masterpiece.")
    
    st.markdown("---")
    
    # Style Selector
    selected_style_name = st.selectbox(
        "Choose an Artistic Style",
        list(STYLES.keys())
    )
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    
    # Run Button placeholder (we check validation first)
    run_btn = st.button("Generate Artwork 🚀")

# Main Content Area
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.title("Neural Art Studio")
    st.markdown("#### Transform ordinary photos into fine art using Deep Learning.")

if uploaded_file is None:
    # Empty State - Beautiful Placeholder
    st.info("👈 Please upload an image from the sidebar to get started.")
    st.markdown("""
    <div style="text-align: center; padding: 50px; opacity: 0.5;">
        <h1>🖼️</h1>
        <p>Waiting for your creativity...</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # 1. Load Content Image
    content_image = Image.open(uploaded_file).convert('RGB')
    
    # 2. Setup Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(content_image, use_container_width=True)

    # 3. Process Logic
    if run_btn:
        model_path = STYLES[selected_style_name]
        
        with st.spinner(f"Painting in the style of {selected_style_name}..."):
            style_model = load_model(model_path)
            
            if style_model is None:
                st.error(f"❌ Could not load model: {model_path}. Check your files.")
            else:
                output_image = stylize(content_image, style_model)
                
                with col2:
                    st.subheader("Masterpiece")
                    st.image(output_image, use_container_width=True)
                    
                    # Download Button
                    buf = io.BytesIO()
                    output_image.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download High-Res 📥",
                        data=byte_im,
                        file_name="masterpiece.jpg",
                        mime="image/jpeg"
                    )
                
                st.success("Transformation Complete!")