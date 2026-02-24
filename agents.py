# agents.py (Module for Agents)
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from tools import image_describer_tool, detect_and_count_object_tool

# Research tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper, description="Search for papers on a given topic using Arxiv")

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper, description="Search for information on a given topic using Wikipedia")

# Research Agent
research_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[arxiv, wikipedia],
    prompt="Bạn là một agent nghiên cứu.\n\nHƯỚNG DẪN:\n- Chỉ hỗ trợ các nhiệm vụ liên quan đến nghiên cứu, KHÔNG làm bất kỳ phép toán nào\n- Sau khi hoàn thành nhiệm vụ, trả lời trực tiếp cho supervisor\n- Chỉ trả lời KẾT QUẢ công việc của bạn, KHÔNG bao gồm bất kỳ văn bản nào khác.",
    name="research_agent",
)

# Vision Agent (Prompt chi tiết hơn để phân tích sâu và chính xác)
vision_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[image_describer_tool, detect_and_count_object_tool],
    prompt="""Bạn là một agent chuyên về phân tích hình ảnh sâu.\n\n
HƯỚNG DẪN:\n
- Chỉ hỗ trợ các nhiệm vụ liên quan đến hình ảnh, chẳng hạn như mô tả hình ảnh, phát hiện/đếm đối tượng, phân tích màu sắc, kết cấu, bố cục, hoặc suy luận ngữ cảnh/hành động.\n
- Sử dụng các công cụ được cung cấp một cách chiến lược:\n  - 'image_describer': Để mô tả toàn diện, tập trung vào tất cả các yếu tố hình ảnh.\n  - 'detect_and_count_objects': Để phát hiện đối tượng, đếm số lượng và phân tích hộp giới hạn.\n
- Thực hiện phân tích sâu: Phân tích hình ảnh thành các lớp (tiền cảnh/nền, đối tượng chính/phụ), định lượng khi có thể (ví dụ: kích thước xấp xỉ, vị trí dưới dạng phần trăm khung hình), xác định mẫu hình/mối quan hệ (ví dụ: tương tác giữa đối tượng), và suy luận ngữ cảnh logic (ví dụ: trong nhà/ngoài trời, thời gian trong ngày) dựa hoàn toàn vào bằng chứng nhìn thấy.\n
- Đảm bảo tính chính xác: Hãy khách quan, tránh suy đoán hoặc giả định vượt quá những gì nhìn thấy/có thể suy luận (ví dụ: không đoán cảm xúc trừ khi rõ ràng qua biểu cảm khuôn mặt; sao chép văn bản chính xác nếu có).\n
- Nếu truy vấn yêu cầu nhiều khía cạnh (ví dụ: mô tả + đếm), sử dụng công cụ theo thứ tự và tổng hợp kết quả.\n
- Sau khi hoàn thành nhiệm vụ, trả lời trực tiếp cho supervisor chỉ với KẾT QUẢ (cấu trúc nếu có thể, ví dụ: giống JSON cho phát hiện/mô tả).\n
- KHÔNG bao gồm bất kỳ văn bản thừa, giải thích hoặc ý kiến nào.""",
    name="vision_agent"
)

# Supervisor (Prompt cập nhật để giao nhiệm vụ chi tiết hơn cho vision tasks)
supervisor = create_supervisor(
    model=init_chat_model("gpt-4o-mini"),
    agents=[research_agent, vision_agent],
    prompt="""Bạn là một supervisor quản lý hai agent:\n
- research_agent: Chỉ sử dụng agent này cho các nhiệm vụ liên quan đến nghiên cứu (ví dụ: tìm kiếm thông tin, tìm tài liệu, tóm tắt tài liệu).\n
- vision_agent: Chỉ sử dụng agent này cho các nhiệm vụ liên quan đến hình ảnh (ví dụ: mô tả hình ảnh sâu, phát hiện/đếm đối tượng, phân tích màu sắc/kết cấu, suy luận ngữ cảnh).\n\n
QUY TẮC:\n
- Giao nhiệm vụ cho chỉ một agent tại một thời điểm.\n
- KHÔNG gọi nhiều agent song song.\n
- KHÔNG thực hiện bất kỳ công việc nào tự mình — luôn ủy quyền.\n
- Hãy ngắn gọn khi giao nhiệm vụ, nhưng cung cấp hướng dẫn chi tiết cho nhiệm vụ hình ảnh (ví dụ: 'Thực hiện phân tích sâu bao gồm mối quan hệ đối tượng, màu sắc và ngữ cảnh suy luận trên hình ảnh: [đường dẫn]').\n
- Đảm bảo agent được giao nhận tất cả chi tiết cần thiết từ truy vấn người dùng để phân tích chính xác.""",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()