import streamlit as st
import os
from pipeline import AvatarExtractorPipeline

st.set_page_config(page_title="3D Avatar Extractor", layout="wide")

st.title("🧍 3D Avatar Extractor")
st.write("Upload an image and we will extract your 3D avatar (`.obj`).")

@st.cache_resource
def get_pipeline():
    st.info("Downloading and initializing SAM-3D-Body...")
    return AvatarExtractorPipeline()

try:
    pipeline = get_pipeline()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Predefined Skin Tones Mapping for Frontend Coloring 
skin_tones = {
    "Fair": "#FDF1E8",
    "Light": "#F3D8C4",
    "Medium": "#D5AC8A",
    "Tan": "#BB8D6A",
    "Coco": "#7E4E30",
    "Deep": "#3D2314"
}

with st.sidebar:
    st.header("Avatar Settings")
    selected_tone_name = st.selectbox("Select Skin Tone Category:", list(skin_tones.keys()), index=2)

selected_color_hex = skin_tones[selected_tone_name]

uploaded_file = st.file_uploader("Upload Full Body Picture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    temp_img_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Photo")
        st.image(temp_img_path, use_column_width=True)

    with st.spinner("Extracting 3D Avatar..."):
        try:
            mesh_path = pipeline.extract_avatar(temp_img_path)
            
            with col2:
                st.header("🧑‍🦲 3D Avatar")
                
                # Render 3D Model Interactively
                import trimesh
                import plotly.graph_objects as go
                
                mesh = trimesh.load(mesh_path)
                vertices = mesh.vertices
                faces = mesh.faces
                
                fig = go.Figure(data=[go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 2],   # Native depth mapped to Plotly Y
                    z=-vertices[:, 1],  # Native downward Y inverted and mapped to Plotly Up-Z
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=selected_color_hex,
                    opacity=1.0,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5)
                )])
                
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        aspectmode='data'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                with open(mesh_path, "rb") as f:
                    st.download_button("Download 3D Model (.obj)", data=f, file_name="avatar.obj", mime="application/octet-stream")
                st.success("Avatar Extraction Complete!")

        except Exception as e:
            st.error(f"Error processing image: {e}")
