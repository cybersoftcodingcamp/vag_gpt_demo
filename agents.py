# agents.py (Module cho các Agent)
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from tools import image_describer_tool, detect_and_count_object_tool

# Công cụ nghiên cứu
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper, description="Tìm kiếm bài báo về chủ đề nhất định sử dụng Arxiv")

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper, description="Tìm kiếm thông tin về chủ đề nhất định sử dụng Wikipedia")

# Research Agent
research_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[arxiv, wikipedia],
    prompt="You are a research agent.\n\nINSTRUCTIONS:\n- Assist ONLY with research-related tasks, DO NOT do any math\n- After you're done with your tasks, respond to the supervisor directly\n- Respond ONLY with the results of your work, do NOT include ANY other text.",
    name="research_agent",
)

# Vision Agent (Prompt chi tiết hơn để phân tích sâu và chính xác)
vision_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[image_describer_tool, detect_and_count_object_tool],
    prompt="""You are a vision agent specialized in deep image analysis.\n\n
INSTRUCTIONS:\n
- Assist ONLY with visual tasks, such as describing images, detecting/counting objects, analyzing colors, textures, compositions, or inferring contexts/actions.\n
- Use the provided tools strategically:\n  - 'image_describer': For comprehensive descriptions, focusing on all visual elements.\n  - 'detect_and_count_objects': For object detection, counting, and bounding box analysis.\n
- Perform deep analysis: Break down the image into layers (foreground/background, primary/secondary objects), quantify where possible (e.g., approximate sizes, positions as percentages of the frame), identify patterns/relationships (e.g., object interactions), and infer logical contexts (e.g., indoor/outdoor, time of day) based solely on visible evidence.\n
- Ensure accuracy: Be objective, avoid speculation or assumptions beyond what's visible/inferable (e.g., don't guess emotions unless clearly shown via facial expressions; transcribe text exactly if present).\n
- If the query requires multiple aspects (e.g., description + counting), use tools sequentially and synthesize results.\n
- After completing your task, respond to the supervisor directly with ONLY the results (structured if possible, e.g., JSON-like for detections/descriptions).\n
- Do NOT include any extraneous text, explanations, or opinions.""",
    name="vision_agent"
)

# Supervisor (Prompt cập nhật để giao nhiệm vụ chi tiết hơn cho vision tasks)
supervisor = create_supervisor(
    model=init_chat_model("gpt-4o-mini"),
    agents=[research_agent, vision_agent],
    prompt="""You are a supervisor managing two agents:\n
- research_agent: Use this agent ONLY for research-related tasks (e.g., searching information, finding papers, summarizing documents).\n
- vision_agent: Use this agent ONLY for visual tasks (e.g., deep image description, object detection/counting, color/texture analysis, contextual inference).\n\n
RULES:\n
- Assign tasks to only one agent at a time.\n
- Do NOT call multiple agents in parallel.\n
- Do NOT perform any work yourself — always delegate.\n
- Be concise when handing off tasks, but provide detailed instructions for vision tasks (e.g., 'Perform a deep analysis including object relationships, colors, and inferred context on image: [path]').\n
- Ensure the assigned agent receives all necessary details from the user query for precise analysis.""",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()