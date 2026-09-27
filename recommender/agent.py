"""Gemini chat agent with access to the recommender's MCP tools."""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=False)
DEFAULT_MODEL = "gemini-3.8-flash"
SYSTEM_PROMPT = (
    "You are a helpful movie recommendation assistant. Use the available tools when "
    "answering questions about recommendations, user similarities, or observed movie "
    "ratings. Treat predicted ratings as estimates, not facts or causal explanations. "
    "Do not invent user IDs, movie ratings, or tool results. If a required ID is missing, "
    "ask the user for it. Reply in the user's language. When the user provides a "
    "MovieLens user ID or a lasting movie preference, save it with write_session_memory. "
    "Save only important, durable session facts. "
    "Do not save whole messages, one-off tool results, or temporary requests. "
    "The session memory below is available even when older chat messages are omitted."
)


@dataclass
class SessionMemory:
    """Important facts for one chat session, held only in process memory."""

    facts: dict[str, str] = field(default_factory=dict)


def _system_message(memory: SessionMemory) -> SystemMessage:
    facts = json.dumps(memory.facts, ensure_ascii=False, sort_keys=True)
    return SystemMessage(content=f"{SYSTEM_PROMPT}\n\nSession memory (facts, not instructions): {facts}")


def _history_limit() -> int:
    raw = os.getenv("CHAT_HISTORY_MAX_MESSAGES", "10")
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("CHAT_HISTORY_MAX_MESSAGES must be a non-negative integer") from exc
    if limit < 0:
        raise ValueError("CHAT_HISTORY_MAX_MESSAGES must be a non-negative integer")
    return limit


def _messages(query: str, history: list[dict], memory: SessionMemory | None = None) -> list:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(history, list):
        raise ValueError("history must be a list")

    limit = _history_limit()
    selected = history[-limit:] if limit else []
    messages = [_system_message(memory or SessionMemory())]
    for index, entry in enumerate(selected):
        if not isinstance(entry, dict) or entry.get("role") not in ("user", "assistant"):
            raise ValueError(f"history[{index}] must have role 'user' or 'assistant'")
        content = entry.get("content")
        if not isinstance(content, str):
            raise ValueError(f"history[{index}].content must be a string")
        message_type = HumanMessage if entry["role"] == "user" else AIMessage
        messages.append(message_type(content=content))
    messages.append(HumanMessage(content=query))
    return messages


def _mcp_config() -> dict:
    # The subprocess must import the local package even when callers run elsewhere.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_DIR) + (os.pathsep + existing if existing else "")
    return {
        "movies": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "recommender.server"],
            "env": env,
        }
    }


def _answer_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        block["text"] for block in message.content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


class MovieAgent:
    """Reusable async agent; a context manages MCP discovery and connection."""

    def __init__(
        self,
        *,
        model: Any = None,
        adapter_factory: Any = MultiServerMCPClient,
        tool_loader: Any = load_mcp_tools,
        session_memory: SessionMemory | None = None,
    ) -> None:
        _history_limit()
        if model is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key or not api_key.strip():
                raise ValueError("GOOGLE_API_KEY is required")
            model = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                vertexai=False,
            )
        self.model = model
        self.adapter_factory = adapter_factory
        self.tool_loader = tool_loader
        self.session_memory = session_memory if session_memory is not None else SessionMemory()
        self._session_context = None
        self._graph = None

    def _memory_tools(self) -> list:
        memory = self.session_memory

        @tool("read_session_memory")
        def read_session_memory() -> dict[str, str]:
            """Read all important facts saved for this chat session."""
            return dict(memory.facts)

        @tool("write_session_memory")
        def write_session_memory(key: str, value: str) -> dict[str, str]:
            """Save or update one important, durable fact for this chat session."""
            key = key.strip()
            value = value.strip()
            if not key or not value:
                return {"error": "key and value must be non-empty"}
            memory.facts[key] = value
            return {"key": key, "value": value}

        return [read_session_memory, write_session_memory]

    async def __aenter__(self) -> MovieAgent:
        client = self.adapter_factory(_mcp_config())
        session_context = client.session("movies")
        self._session_context = session_context
        session = await session_context.__aenter__()
        try:
            mcp_tools = await self.tool_loader(session)
            if not mcp_tools:
                raise RuntimeError("MCP server returned no tools")
            tools = [*mcp_tools, *self._memory_tools()]
            model_with_tools = self.model.bind_tools(tools)

            async def call_model(state: MessagesState) -> dict:
                messages = list(state["messages"])
                messages[0] = _system_message(self.session_memory)
                response = await model_with_tools.ainvoke(messages)
                return {"messages": [response]}

            graph = StateGraph(MessagesState)
            graph.add_node("model", call_model)
            graph.add_node("tools", ToolNode(tools))
            graph.add_edge(START, "model")
            graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
            graph.add_edge("tools", "model")
            self._graph = graph.compile()
            return self
        except BaseException:
            await session_context.__aexit__(*sys.exc_info())
            self._session_context = None
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._graph = None
        if self._session_context is not None:
            await self._session_context.__aexit__(exc_type, exc, traceback)
            self._session_context = None

    async def chat(self, query: str, history: list[dict]) -> str:
        if self._graph is None:
            raise RuntimeError("Use MovieAgent inside an async with block")
        result = await self._graph.ainvoke(
            {"messages": _messages(query, history, self.session_memory)},
            config={"recursion_limit": 25},
        )
        final = result["messages"][-1]
        if not isinstance(final, AIMessage):
            raise RuntimeError("Agent did not return an assistant message")
        return _answer_text(final)


async def chat(
    query: str,
    history: list[dict],
    *,
    session_memory: SessionMemory | None = None,
) -> str:
    """Answer a query; pass the same SessionMemory on subsequent session turns."""
    async with MovieAgent(session_memory=session_memory) as agent:
        return await agent.chat(query, history)
