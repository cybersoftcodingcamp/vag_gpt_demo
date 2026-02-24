# app.py (Ứng dụng Streamlit chính)
import streamlit as st
import os
import tempfile
from PIL import Image
import requests
from dotenv import load_dotenv
from agents import supervisor

# Tải biến môi trường
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Cấu hình UI
st.set_page_config(
    page_title="Visual Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 8px; height: 3em;}
    .uploadedFile {border: 2px dashed #0066cc; border-radius: 10px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# Thanh bên
with st.sidebar:
    st.image("./images/vag-logo.png", caption="Visual Agentic AI")
    st.header("🤖 Visual Agentic AI")
    st.markdown("""
    **Tính năng:**
    - Tải lên ảnh hoặc dán liên kết ảnh
    - Hỏi bất kỳ điều gì về ảnh (mô tả, đếm vật, màu sắc…)
    - Tìm kiếm nghiên cứu khoa học liên quan
    - Supervisor tự động phân công Research / Vision Agent
    """)
    
    st.divider()
    st.header("📖 Hướng dẫn")
    st.markdown("""
    1. Chọn cách đưa ảnh vào (Tải lên hoặc URL)  
    2. Nhập câu hỏi của bạn  
    3. Nhấn **Phân tích ảnh**
    """)

# Nội dung chính
st.title("🤖 Visual Agentic AI – Cybersoft AI")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📸 Ảnh đầu vào")
    input_mode = st.radio("Chọn cách đưa ảnh vào:", 
                         ["Tải lên ảnh từ máy", "Dán liên kết URL"], 
                         horizontal=True)

    image_source = None
    preview_image = None

    if input_mode == "Tải lên ảnh từ máy":
        uploaded_file = st.file_uploader(
            "Chọn ảnh (PNG, JPG, JPEG, WEBP)",
            type=["png", "jpg", "jpeg", "webp"],
            help="Tối đa 10MB"
        )
        if uploaded_file:
            # Lưu tạm vào đĩa để công cụ có thể đọc
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.getbuffer())
                image_source = tmp.name
            preview_image = Image.open(uploaded_file)
            st.success("Ảnh đã tải lên thành công!")

    else:  # Chế độ URL
        url = st.text_input("Dán liên kết ảnh:", placeholder="https://example.com/image.jpg")
        if url:
            image_source = url.strip()
            try:
                preview_image = Image.open(requests.get(url, stream=True).raw)
                st.success("Liên kết hợp lệ!")
            except:
                st.error("Không thể tải ảnh từ liên kết này")

    # Xem trước
    if preview_image:
        st.image(preview_image, caption="Ảnh xem trước", use_column_width=True)

with col2:
    st.subheader("❓ Câu hỏi của bạn")
    query = st.text_area(
        "Nhập câu hỏi (ví dụ: \"Mô tả ảnh này\", \"Có bao nhiêu con chó?\", \"Màu lông con chó là gì? Tìm thêm thông tin về giống chó này\")",
        height=120,
        placeholder="Hỏi gì cũng được về ảnh..."
    )

    if st.button("🚀 Phân tích ảnh", type="primary", use_container_width=True):
        if not query:
            st.warning("Vui lòng nhập câu hỏi!")
        elif not image_source:
            st.warning("Vui lòng tải lên ảnh hoặc dán liên kết!")
        else:
            with st.spinner("Đang xử lý... Supervisor đang phân công agent..."):
                # Tạo câu hỏi đầy đủ cho hệ thống
                full_query = f"{query}\nImage: {image_source}"

                output_container = st.empty()
                full_response = ""

                try:
                    for chunk in supervisor.stream(
                        {"messages": [{"role": "user", "content": full_query}]}
                    ):
                        messages = chunk.get("supervisor", {}).get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            # Lấy nội dung văn bản
                            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                            full_response += content + "\n"
                            output_container.markdown(full_response)

                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

                # Xóa file tạm sau khi xong
                if input_mode == "Tải lên ảnh từ máy" and image_source and os.path.exists(image_source):
                    os.unlink(image_source)