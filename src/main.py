import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from psycopg_pool import ConnectionPool
from pydantic import BaseModel
# from redis import asyncio as aioredis

from src.category_definition import define_category
from src.init_tools import tools
from src.logger import LOGGER

load_dotenv(override=True)

LOGGER.info("BEGIN")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global db_connection
    global agent

    # redis = aioredis.from_url("redis://redis_cache:6379")
    # FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

    # try:
    #     await redis.ping()
    #     LOGGER.info("Successfully connected to Redis")
    # except Exception as e:
    #     LOGGER.error(f"Redis connection failed: {e}")

    DB_URI = f"postgresql://{os.getenv('PERSISTENCE_PG_USER')}:{os.getenv('PERSISTENCE_PG_PASSWORD')}@{os.getenv('PERSISTENCE_PG_CONTAINER')}:{os.getenv('PERSISTENCE_PG_PORT')}/{os.getenv('PERSISTENCE_PG_DB')}?sslmode=disable"
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    db_connection = ConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    )
    checkpointer = PostgresSaver(db_connection)
    checkpointer.setup()

    agent_llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o-2024-08-06",
        streaming=False,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = os.getenv("agent_prefix_prompt")
    system_message = SystemMessage(content=prompt)

    agent = create_react_agent(
        agent_llm, tools=tools, state_modifier=system_message, checkpointer=checkpointer
    )

    yield

    if db_connection:
        db_connection.close()
    # await redis.close()


app = FastAPI(lifespan=lifespan)

cache_map = {}


class Query(BaseModel):
    conversation_id: str
    text: str
    # cache_enabled: bool


db_connection_pool = None
agent = None


def composite_key_builder(func, *args, **kwargs):
    query_text = kwargs.get("query").text
    conversation_id = kwargs.get("query").conversation_id
    key = f"{func.__name__}:{conversation_id}:{query_text}"
    LOGGER.info(f"Generated cache key: {key}")
    return key


def transform_response_format(json_new_format):
    """
    Переводит ответ langgraph агента (новый формат) в формат langchain (старый). В новом формате отсутствуют индивидуальные мысли по инструментам.
    """
    messages_list = json_new_format["messages"]
    actual_messages = messages_list
    for i in range(len(messages_list) - 1, -1, -1):
        if isinstance(messages_list[i], HumanMessage):
            actual_messages = messages_list[i:]
            break

    LOGGER.info(f"ACTUAL MESSAGES: {actual_messages}")

    input_value = actual_messages[0].content
    output_value = actual_messages[-1].content

    intermediate_steps = []
    for message in actual_messages:
        if (
            isinstance(message, AIMessage)
            and len(getattr(message, "tool_calls", [])) > 0
        ):
            for tool_call in message.tool_calls:
                tool_output = None
                for next_message in actual_messages:
                    if isinstance(next_message, ToolMessage):
                        if next_message.tool_call_id == tool_call["id"]:
                            tool_output = next_message.content
                            break

                step = {
                    "type": "AgentAction",
                    "thought": message.content,
                    "tool": tool_call["name"],
                    "tool_input": tool_call["args"]["query"],
                    "tool_output": tool_output,
                }
                intermediate_steps.append(step)

    json_old_format = {
        "input": input_value,
        "output": output_value,
        "intermediate_steps": intermediate_steps,
    }

    return json_old_format


@app.post("/chat")
async def chat(query: Query = Body(...)):
    try:
        return await asyncio.wait_for(_execute_chat_logic(query), timeout=150)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")


async def _execute_chat_logic(
    query: Query = Body(...),
):
    try:
        global agent
        LOGGER.info(f"Query: {query.text}")

        # cache_key = composite_key_builder(chat, query=query)

        # cache_backend = FastAPICache.get_backend()

        # cached_response = await cache_backend.get(cache_key)
        # if cached_response:
        #     LOGGER.info("Returning cached response")
        #     return json.loads(cached_response)

        config = {"configurable": {"thread_id": query.conversation_id}}
        input_message = HumanMessage(content=query.text)
        response = agent.invoke({"messages": [input_message]}, config)

        LOGGER.info(response)

        response = transform_response_format(dict(response))
        response["query_category"] = define_category(query.text)

        # response["cache_enabled"] = True

        LOGGER.info(f"Response: {response}")

        # delete tool calls messages
        messages = agent.get_state(config).values["messages"]
        tools_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]

        ai_tool_calls = [
            message
            for message in messages
            if type(message) is AIMessage
            and len(getattr(message, "tool_calls", [])) > 0
        ]

        delete_messages = tools_messages + ai_tool_calls

        for message in delete_messages:
            agent.update_state(config, {"messages": RemoveMessage(id=message.id)})

        # delete old messages
        messages = agent.get_state(config).values["messages"]
        # TODO вынести -10 в конфиг (это 5 последних вопросов пользователя)
        early_messages = messages[:-10]
        for message in early_messages:
            agent.update_state(config, {"messages": RemoveMessage(id=message.id)})

        # await cache_backend.set(cache_key, json.dumps(response), expire=300)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")


@app.get("/health")
async def health():
    try:
        return await asyncio.wait_for(_execute_health_logic(), timeout=10)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")


async def _execute_health_logic():
    return "OK"
