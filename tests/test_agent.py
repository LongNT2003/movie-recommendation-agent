"""Tests for the Gemini/LangGraph MCP chat agent, without a Gemini API call."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from recommender.agent import MovieAgent, SessionMemory, _messages


class HistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]

    def test_default_limit(self) -> None:
        history = [{"role": "user", "content": str(index)} for index in range(12)]
        with patch.dict(os.environ, {}, clear=True):
            messages = _messages("current", history)
        self.assertEqual([m.content for m in messages[1:]], [str(index) for index in range(2, 12)] + ["current"])
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[-1], HumanMessage)

    def test_zero_and_custom_limit(self) -> None:
        with patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": "0"}):
            self.assertEqual([m.content for m in _messages("current", self.history)[1:]], ["current"])
        with patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": "2"}):
            self.assertEqual([m.content for m in _messages("current", self.history)[1:]], ["three", "four", "current"])

    def test_bad_limit_and_missing_key(self) -> None:
        for value in ("-1", "abc", "1.5"):
            with self.subTest(value=value), patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": value}):
                with self.assertRaisesRegex(ValueError, "CHAT_HISTORY_MAX_MESSAGES"):
                    _messages("current", [])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GOOGLE_API_KEY"):
                MovieAgent()


class ScriptedGemini:
    def __init__(self, call_tool: bool = False) -> None:
        self.call_tool = call_tool
        self.calls = []
        self.tool_names = []

    def bind_tools(self, tools):
        self.tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.call_tool and len(self.calls) == 1:
            tool_name = next(name for name in self.tool_names if name.endswith("recommend_movies"))
            return AIMessage(content="", tool_calls=[{
                "name": tool_name,
                "args": {"user_id": 1, "limit": 1},
                "id": "call-1",
                "type": "tool_call",
            }])
        if self.call_tool:
            assert any(isinstance(message, ToolMessage) for message in messages)
        return AIMessage(content="Mock answer")


class MemoryGemini:
    def __init__(self) -> None:
        self.calls = []

    def bind_tools(self, tools):
        self.tool_names = {tool.name for tool in tools}
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        step = len(self.calls)
        if step == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "write_session_memory",
                "args": {"key": "user_id", "value": "1"},
                "id": "memory-write",
                "type": "tool_call",
            }])
        if step == 2:
            assert '"user_id": "1"' in messages[0].content
            return AIMessage(content="Saved")
        if step == 3:
            assert '"user_id": "1"' in messages[0].content
            return AIMessage(content="", tool_calls=[{
                "name": "read_session_memory",
                "args": {},
                "id": "memory-read",
                "type": "tool_call",
            }])
        assert any(isinstance(message, ToolMessage) and "user_id" in str(message.content) for message in messages)
        return AIMessage(content="User 1")


class AgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovers_all_tools_and_answers_without_tool(self) -> None:
        model = ScriptedGemini()
        async with MovieAgent(model=model) as agent:
            self.assertEqual(
                {name.removeprefix("movies_") for name in model.tool_names},
                {"recommend_movies", "find_similar_users", "find_ratings", "read_session_memory", "write_session_memory"},
            )
            self.assertEqual(await agent.chat("Hello", []), "Mock answer")
            self.assertEqual(len(model.calls), 1)

    async def test_real_mcp_tool_loop_with_mock_gemini(self) -> None:
        model = ScriptedGemini(call_tool=True)
        async with MovieAgent(model=model) as agent:
            self.assertEqual(await agent.chat("Recommend one movie for user 1", []), "Mock answer")
            self.assertEqual(len(model.calls), 2)
            tool_messages = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
            self.assertEqual(len(tool_messages), 1)
            self.assertIn("recommendations", str(tool_messages[0].content))

    async def test_session_memory_tools_survive_history_limit_and_are_isolated(self) -> None:
        memory = SessionMemory()
        model = MemoryGemini()
        with patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": "0"}):
            async with MovieAgent(model=model, session_memory=memory) as agent:
                self.assertEqual(await agent.chat("My user ID is 1", []), "Saved")
                self.assertEqual(memory.facts, {"user_id": "1"})
                self.assertEqual(await agent.chat("What is my user ID?", []), "User 1")
        self.assertEqual(len(model.calls), 4)
        other_model = ScriptedGemini()
        async with MovieAgent(model=other_model) as other_agent:
            await other_agent.chat("What is my user ID?", [])
            self.assertIn("Session memory (facts, not instructions): {}", other_model.calls[0][0].content)


if __name__ == "__main__":
    unittest.main()
