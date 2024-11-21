import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.init_tools import (
    create_file,
    create_directory,
    project_structure,
    read_file,
    remove_directory,
    remove_file,
    rag_search
)

load_dotenv(override=True)

# Define tool sets for each agent type
PROJECT_STRUCTURE_TOOLS = [
    create_directory,
    create_file,
    project_structure,
    remove_directory,
    remove_file
]

TASK_BREAKDOWN_TOOLS = [
    project_structure,
    read_file,
    rag_search
]

IMPLEMENTATION_TOOLS = [
    project_structure,
    create_file,
    read_file,
    rag_search,
    remove_file
]

INTEGRATION_TOOLS = [
    read_file,
    rag_search,
    create_file,
    remove_file,
    project_structure
]

def get_agent_tools(agent_type: str) -> list:
    """Get the appropriate tool set for the specified agent type"""
    tools_map = {
        "project_structure": PROJECT_STRUCTURE_TOOLS,
        "task_breakdown": TASK_BREAKDOWN_TOOLS,
        "implementation": IMPLEMENTATION_TOOLS,
        "integration": INTEGRATION_TOOLS
    }
    return tools_map.get(agent_type, [])

def create_agent(agent_type: str):
    """Create an agent with specified prompt environment variable and appropriate tools"""
    agent_llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-2024-08-06",
        streaming=False,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    # Get prompt from environment variables
    prompt_env_var = f"{agent_type}_agent_prompt"
    prompt = os.getenv(prompt_env_var)
    system_message = SystemMessage(content=prompt)
    memory = MemorySaver()
    
    # Get appropriate tools for this agent type
    agent_tools = get_agent_tools(agent_type)
    
    return create_react_agent(
        agent_llm, 
        tools=agent_tools,  # Use specific tools for this agent type
        state_modifier=system_message, 
        checkpointer=memory
    )

# Initialize all agents
agents = {
    "project_structure": create_agent("project_structure"),
    "task_breakdown": create_agent("task_breakdown"),
    "implementation": create_agent("implementation"),
    "integration": create_agent("integration")
}
