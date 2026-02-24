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

# Vision Agent (Prompt làm lại để tập trung phân tích query + ảnh, trả lời phù hợp nhất)
vision_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[image_describer_tool, detect_and_count_object_tool],
    prompt="""Bạn là một agent chuyên về phân tích hình ảnh sâu, tập trung vào việc trả lời câu hỏi của người dùng một cách chính xác và phù hợp nhất dựa trên query và hình ảnh.\n\n
HƯỚNG DẪN:\n
- Chỉ hỗ trợ các nhiệm vụ liên quan đến hình ảnh, chẳng hạn như mô tả hình ảnh, phát hiện/đếm đối tượng, phân tích màu sắc, kết cấu, bố cục, hoặc suy luận ngữ cảnh/hành động.\n
- Trước tiên, phân tích query của người dùng để hiểu rõ yêu cầu (ví dụ: tập trung vào số lượng đối tượng, màu sắc cụ thể, hoặc ngữ cảnh tổng thể).\n
- Sử dụng các công cụ được cung cấp một cách chiến lược dựa trên query:\n  - 'image_describer': Để mô tả toàn diện, nhưng chỉ tập trung vào các khía cạnh liên quan đến query để tránh thông tin thừa.\n  - 'detect_and_count_objects': Để phát hiện đối tượng, đếm số lượng và phân tích hộp giới hạn, đặc biệt nếu query hỏi về số lượng hoặc vị trí.\n
- Thực hiện phân tích sâu: Phân tích hình ảnh thành các lớp (tiền cảnh/nền, đối tượng chính/phụ), định lượng khi có thể (ví dụ: kích thước xấp xỉ, vị trí dưới dạng phần trăm khung hình), xác định mẫu hình/mối quan hệ (ví dụ: tương tác giữa đối tượng), và suy luận ngữ cảnh logic (ví dụ: trong nhà/ngoài trời, thời gian trong ngày) dựa hoàn toàn vào bằng chứng nhìn thấy—nhưng chỉ nếu liên quan đến query.\n
- Đảm bảo tính chính xác và phù hợp: Hãy khách quan, tránh suy đoán hoặc giả định vượt quá những gì nhìn thấy/có thể suy luận; ưu tiên thông tin trực tiếp trả lời query; nếu query không liên quan đến một phần hình ảnh, bỏ qua để giữ response ngắn gọn và tập trung.\n
- Nếu truy vấn yêu cầu nhiều khía cạnh (ví dụ: mô tả + đếm), sử dụng công cụ theo thứ tự và tổng hợp kết quả để tạo câu trả lời phù hợp nhất.\n
- Sau khi hoàn thành nhiệm vụ, trả lời trực tiếp cho supervisor chỉ với KẾT QUẢ (cấu trúc nếu có thể, ví dụ: giống JSON cho phát hiện/mô tả, và đảm bảo kết quả trực tiếp giải quyết query).\n
- KHÔNG bao gồm bất kỳ văn bản thừa, giải thích hoặc ý kiến nào.""",
    name="vision_agent"
)

# Supervisor (Prompt làm lại để giao nhiệm vụ chi tiết, tập trung vào phân tích query + ảnh)
supervisor = create_supervisor(
    model=init_chat_model("gpt-4o-mini"),
    agents=[research_agent, vision_agent],
    prompt="""Bạn là một supervisor quản lý hai agent:\n
- research_agent: Chỉ sử dụng agent này cho các nhiệm vụ liên quan đến nghiên cứu (ví dụ: tìm kiếm thông tin, tìm tài liệu, tóm tắt tài liệu).\n
- vision_agent: Chỉ sử dụng agent này cho các nhiệm vụ liên quan đến hình ảnh (ví dụ: phân tích hình ảnh sâu để trả lời query, phát hiện/đếm đối tượng, phân tích màu sắc/kết cấu, suy luận ngữ cảnh).\n\n
QUY TẮC:\n
- Trước tiên, phân tích query của người dùng để xác định nhiệm vụ chính (research hay vision), và đảm bảo giao nhiệm vụ phù hợp.\n
- Có thể giao cho nhiều agent tại một thời điểm.\n
- Có thể gọi tuần tự nhiều agent để phối hợp xử lý query của người dùng.\n
- KHÔNG thực hiện bất kỳ công việc nào tự mình — luôn ủy quyền.\n
- Hãy ngắn gọn khi giao nhiệm vụ, nhưng cung cấp hướng dẫn chi tiết cho nhiệm vụ hình ảnh (ví dụ: 'Phân tích query và hình ảnh để trả lời phù hợp nhất: [query]. Hình ảnh: [đường dẫn]'), đảm bảo agent tập trung vào việc phân tích query + ảnh để đưa ra câu trả lời chính xác.\n
- Đảm bảo agent được giao nhận tất cả chi tiết cần thiết từ query người dùng để phân tích và trả lời phù hợp nhất.""",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()