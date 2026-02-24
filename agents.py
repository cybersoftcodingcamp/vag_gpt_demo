# agents.py (Module for Agents)
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from tools import image_describer_tool, detect_and_count_object_tool

# Research tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper, description="Search for papers on a given topic using Arxiv")

wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper, description="Search for information on a given topic using Wikipedia")

# Research Agent
research_agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[arxiv, wikipedia],
    system_prompt="You are a research agent.\n\nINSTRUCTIONS:\n- Assist ONLY with research-related tasks, DO NOT do any math\n- After you're done with your tasks, respond to the supervisor directly\n- Respond ONLY with the results of your work, do NOT include ANY other text.",
    name="research_agent",  # name có thể không được hỗ trợ trực tiếp, nếu lỗi thì bỏ đi hoặc dùng middleware để set
)

# Vision Agent
vision_agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[image_describer_tool, detect_and_count_object_tool],
    system_prompt="You are a vision agent.\n\nINSTRUCTIONS:\n- Assist ONLY with visual tasks (e.g., describing images, detecting and counting objects)\n- Use only the tools provided to analyze visual inputs\n- After completing your task, respond to the supervisor directly\n- Respond ONLY with the results of your work, do NOT include ANY other text.",
    name="vision_agent",  # Tương tự, nếu lỗi thì bỏ
)

# Supervisor
supervisor = create_supervisor(
    model=init_chat_model("gpt-4o-mini"),
    agents=[research_agent, vision_agent],
    prompt="You are a supervisor managing two agents:\n- research_agent: Use this agent ONLY for research-related tasks (e.g., searching information, finding papers, summarizing documents).\n- vision_agent: Use this agent ONLY for visual tasks (e.g., describing images, detecting and counting objects).\n\nRULES:\n- Assign tasks to only one agent at a time.\n- Do NOT call multiple agents in parallel.\n- Do NOT perform any work yourself — always delegate.\n- Be concise when handing off tasks to agents, only provide what is necessary for them to complete the job.",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()