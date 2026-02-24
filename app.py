# app.py (Main Streamlit App)
import streamlit as st
from agents import research_agent, vision_agent, supervisor
from tools import image_describer_tool, detect_and_count_object_tool, encode_image, extractor_chain, yolo_model
from dotenv import load_dotenv

# Set environment variables (thay bằng keys thật của bạn)
import os
# Load variables from .env
load_dotenv()  # Load tất cả từ .env vào os.environ

# Kiểm tra và sử dụng (nếu cần fallback)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Visual Agentic AI", page_icon="🤖", layout="wide")

# CSS tùy chỉnh để làm đẹp
st.markdown("""
    <style>
    .main {background-color: #f0f4f8;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .sidebar .sidebar-content {background-color: #e8f5e9;}
    </style>
""", unsafe_allow_html=True)

# Sidebar: Mô tả dự án và hướng dẫn
with st.sidebar:
    st.image("https://source.unsplash.com/random/300x200/?ai+agent", caption="Visual Agentic AI")  # Hình ảnh placeholder
    st.header("🤖 Visual Agentic AI")
    st.markdown("""
    **Mô tả Dự Án:**  
    Đây là hệ thống AI agentic sử dụng LangGraph để kết hợp:  
    - **Research Agent**: Tìm kiếm thông tin, tài liệu từ Arxiv và Wikipedia.  
    - **Vision Agent**: Phân tích hình ảnh (mô tả, phát hiện và đếm đối tượng sử dụng YOLO).  
    - **Supervisor**: Quản lý và giao nhiệm vụ cho các agent phù hợp.  
    Dự án giúp xử lý query kết hợp research và vision một cách thông minh.
    """)
    
    st.header("📖 Hướng Dẫn Sử Dụng")
    st.markdown("""
    1. Nhập query vào ô text (ví dụ: "What is the latest research on positional embeddings?" hoặc "How many dogs in image: https://example.com/image.jpg").  
    2. Nhấn "Run Query".  
    3. Xem output streaming ở bên dưới.  
    Lưu ý: Query có thể bao gồm URL hình ảnh cho vision tasks.
    """)

# Main content
st.title("🤖 Visual Agentic AI Tester")
st.markdown("Nhập query của bạn dưới đây để test hệ thống.")

query = st.text_area("Query:", height=100, placeholder="Ví dụ: What is the concept visualized in the image? Image: https://huggingface.co/datasets/tmnam20/Storage/resolve/main/rope.png")

if st.button("Run Query"):
    if query:
        with st.spinner("Processing..."):
            output_container = st.empty()
            try:
                for chunk in supervisor.stream({"messages": [{"role": "user", "content": query}]}):
                    messages = chunk.get("supervisor", {}).get("messages", [])
                    if messages:
                        last_message = messages[-1] if isinstance(messages, list) else messages
                        output_container.write(last_message)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Vui lòng nhập query!")