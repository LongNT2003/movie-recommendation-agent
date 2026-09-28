"""Tests for the Gemini/LangGraph MCP chat agent, without a Gemini API call."""

from __future__ import annotations

import os
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from recommender.agent import MovieAgent, PROJECT_DIR, SessionMemory, _messages


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

    def test_history_limit_keeps_entire_turn_with_parallel_tool_results(self):
        turn = [
            HumanMessage(content="Look up two movies"),
            AIMessage(content="", tool_calls=[
                {"name": "lookup", "args": {}, "id": "a", "type": "tool_call"},
                {"name": "lookup", "args": {}, "id": "b", "type": "tool_call"},
            ]),
            ToolMessage(content=[{"type": "text", "text": '{"movie_id": 38}'}], tool_call_id="a", name="lookup"),
            ToolMessage(content="Tool failed", tool_call_id="b", status="error", name="lookup"),
            AIMessage(content="Found one movie"),
        ]
        history = [HumanMessage(content="old"), AIMessage(content="old answer"), *turn]
        with patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": "1"}):
            messages = _messages("Would I like it?", history)
        self.assertEqual(messages[1:-1], turn)
        self.assertEqual(messages[3].tool_call_id, "a")
        self.assertEqual(messages[4].status, "error")
        with patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": "0"}):
            self.assertEqual(len(_messages("current", history)), 2)

    def test_dictionary_tool_history_and_incomplete_history_validation(self):
        call = {"name": "lookup", "args": {}, "id": "a", "type": "tool_call"}
        history = [
            {"role": "user", "content": "Look up a movie"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": '{"movie_id": 38}', "tool_call_id": "a", "name": "lookup"},
            {"role": "assistant", "content": "Found"},
        ]
        with patch.dict(os.environ, {"CHAT_HISTORY_MAX_MESSAGES": "10"}):
            messages = _messages("current", history)
            self.assertIsInstance(messages[3], ToolMessage)
            self.assertEqual(messages[2].tool_calls[0]["id"], messages[3].tool_call_id)
            for broken in ([history[2]], history[:2], [*history[:2], history[3]]):
                with self.subTest(history=broken), self.assertRaisesRegex(ValueError, "tool"):
                    _messages("current", broken)


class ScriptedGemini:
    def __init__(self, call_tool: bool = False, movie_lookup: bool = False) -> None:
        self.call_tool = call_tool
        self.movie_lookup = movie_lookup
        self.calls = []
        self.tool_names = []

    def bind_tools(self, tools):
        self.tool_names = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.call_tool and len(self.calls) == 1:
            suffix = "get_movie_info" if self.movie_lookup else "recommend_movies"
            tool_name = next(name for name in self.tool_names if name.endswith(suffix))
            return AIMessage(content="", tool_calls=[{
                "name": tool_name,
                "args": {"titles": ["Toy Story", "Jumanji"]} if self.movie_lookup else {
                    "user_id": 1, "limit": 1, "include_genres": [], "exclude_genres": ["Animation"]},
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
                {"recommend_movies", "find_similar_users", "find_ratings", "get_movie_info", "read_session_memory", "write_session_memory"},
            )
            self.assertEqual(await agent.chat("Hello", []), "Mock answer")
            self.assertEqual(len(model.calls), 1)

    @unittest.skipUnless(
        (PROJECT_DIR / "artifacts" / "svd_model.npz").exists()
        and (PROJECT_DIR / "artifacts" / "svd_metadata.json").exists(),
        "Recommendation integration requires trained SVD artifacts",
    )
    async def test_real_mcp_tool_loop_with_mock_gemini(self) -> None:
        model = ScriptedGemini(call_tool=True)
        async with MovieAgent(model=model) as agent:
            self.assertEqual(await agent.chat("Recommend one movie for user 1", []), "Mock answer")
            self.assertEqual(len(model.calls), 2)
            tool_messages = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
            self.assertEqual(len(tool_messages), 1)
            self.assertIn("recommendations", str(tool_messages[0].content))
            content = tool_messages[0].content
            payload = json.loads(content if isinstance(content, str) else content[0]["text"])
            self.assertEqual(payload["filters"]["exclude_genres"], ["Animation"])
            self.assertEqual(len(payload["recommendations"]), 1)
            self.assertNotIn("Animation", payload["recommendations"][0]["genres"])

    async def test_movie_lookup_through_real_mcp_and_agent_loop(self) -> None:
        model = ScriptedGemini(call_tool=True, movie_lookup=True)
        history = []
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GOOGLE_API_KEY": ""}):
            trace_path = Path(directory) / "session.jsonl"
            async with MovieAgent(model=model, trace_path=trace_path) as agent:
                self.assertEqual(await agent.chat("Tell me about Toy Story and Jumanji", history), "Mock answer")
                self.assertEqual(len(model.calls), 2)
                tool_messages = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
                self.assertEqual(len(tool_messages), 1)
                content = tool_messages[0].content
                payload = json.loads(content if isinstance(content, str) else content[0]["text"])
                self.assertEqual([item["status"] for item in payload["results"]], ["exact", "exact"])
                self.assertEqual([item["query_title"] for item in payload["results"]], ["Toy Story", "Jumanji"])
                self.assertTrue(all(item["movies"][0]["plot"] for item in payload["results"]))
                self.assertEqual([type(message) for message in history],
                                 [HumanMessage, AIMessage, ToolMessage, AIMessage])
                self.assertEqual(history[1].tool_calls[0]["id"], history[2].tool_call_id)
                self.assertEqual(history[2].content, content)
                self.assertEqual(await agent.chat("Would I like the first movie?", history), "Mock answer")
                self.assertEqual(model.calls[2][1:-1], history[:4])
                self.assertEqual(len(history), 6)
                self.assertEqual(sum(isinstance(message, HumanMessage) for message in history), 2)
                before_failure = list(history)
                with patch.object(agent._graph, "ainvoke", AsyncMock(side_effect=RuntimeError("LLM unavailable"))):
                    with self.assertRaisesRegex(RuntimeError, "LLM unavailable"):
                        await agent.chat("failed turn", history)
                self.assertEqual(history, before_failure)
            records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            events = [record["event"] for record in records]
            self.assertEqual(events, ["user_input", "tool_use", "tool_result", "llm_output",
                                      "user_input", "llm_output", "user_input", "error"])
            tool_start = next(record for record in records if record["event"] == "tool_use")
            tool_end = next(record for record in records if record["event"] == "tool_result")
            self.assertEqual(tool_start["tool_call_id"], tool_end["tool_call_id"])
            self.assertEqual(tool_start["turn_id"], tool_end["turn_id"])
            self.assertEqual(tool_end["result"], payload)
            self.assertTrue(agent.trace.readable_path.exists())

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
