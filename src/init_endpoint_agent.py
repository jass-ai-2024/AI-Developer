import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.init_tools import tools

load_dotenv(override=True)

agent_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-2024-08-06",
    streaming=False,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
prompt = os.getenv("agent_prefix_prompt")

system_message = SystemMessage(content=prompt)

memory = MemorySaver()

agent = create_react_agent(
    agent_llm, tools, state_modifier=system_message, checkpointer=memory
)
