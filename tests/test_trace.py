"""Check readable transcripts, compact tool content and error diagnostics."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

from recommender.movie_search import MovieCatalog, TitleGuesses, guess_titles
from recommender.trace import EventLog, log_event, trace_context


class TraceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "trace.jsonl"
        self.logger = EventLog(self.path)
        csv_path = Path(self.directory.name) / "movies.csv"
        pd.DataFrame([[1, "Toy Story", 1995, "Animation", "CSV plot"]],
                     columns=["movieId", "title", "year", "genres", "plot"]).to_csv(csv_path, index=False)
        self.catalog = MovieCatalog(csv_path)

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    async def test_failure_logs_actual_exception_and_preserves_result(self):
        with trace_context(self.logger, turn_id="turn-1"):
            result = await self.catalog.search(
                ["Toy Story", "it take to"],
                AsyncMock(side_effect=RuntimeError("Structured output validation failed")),
            )
        error = next(record for record in self.records() if record.get("step") == "search.guess.error")
        self.assertEqual(error["error_type"], "RuntimeError")
        self.assertIn("Structured output validation failed", error["error"])
        self.assertNotIn("traceback", error)
        self.assertEqual(error["search_id"], result["search_id"])
        self.assertEqual(error["turn_id"], "turn-1")
        self.assertEqual([item["status"] for item in result["results"]], ["exact", "lookup_error"])

    async def test_transcript_keeps_tool_payload_once_without_history_or_metadata(self):
        with trace_context(self.logger, turn_id="turn-1"):
            log_event("chat.start", query="Toy Stori?", history_count=200)
            log_event("llm.start", messages=["large system prompt", "old history"])
            log_event("tool.start", tool="get_movie_info", tool_call_id="call-1",
                      arguments={"titles": ["Toy Stori"]})
            result = await self.catalog.search(["Toy Stori"], AsyncMock(return_value={
                "guesses": [{"query_index": 0, "titles": ["Toy Story", "Unknown"]}],
            }))
            log_event("tool.end", tool="get_movie_info", tool_call_id="call-1", result=ToolMessage(
                content=[{"type": "text", "text": json.dumps(result)}],
                tool_call_id="call-1", artifact={"duplicate": result},
                additional_kwargs={"metadata": "unneeded"}))
            log_event("history.append", added_messages=["large history"])
            log_event("llm.end", response=AIMessage(content="Found Toy Story"))
            log_event("chat.end", answer="Found Toy Story", session_memory={"user_id": "288"})
        records = self.records()
        self.assertEqual([record["event"] for record in records],
                         ["user_input", "tool_use", "tool_result", "llm_output"])
        self.assertEqual(records[2]["result"], result)
        self.assertEqual(records[2]["result"]["results"][0]["movies"][0]["plot"], "CSV plot")
        readable = self.logger.readable_path.read_text(encoding="utf-8")
        self.assertEqual(readable.count("CSV plot"), 1)
        for label in ("USER INPUT", "TOOL USE", "TOOL RESULT", "LLM OUTPUT"):
            self.assertIn(label, readable)
        for noise in ("large system prompt", "old history", "unneeded", "session_memory", "artifact"):
            self.assertNotIn(noise, readable)
        self.assertIn('\n  "results": [\n', readable)
        self.assertEqual(records[1]["tool_call_id"], records[2]["tool_call_id"])

    async def test_parsing_failure_logs_short_error_without_raw_llm_metadata(self):
        runnable = AsyncMock()
        runnable.ainvoke.return_value = {
            "raw": AIMessage(content="malformed output"), "parsed": None,
            "parsing_error": ValueError("Invalid title schema"),
        }
        with patch("recommender.movie_search.ChatGoogleGenerativeAI") as model, \
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-for-mock"}), trace_context(self.logger):
            model.return_value.with_structured_output.return_value = runnable
            result = await self.catalog.search(["it take to"], guess_titles)
            model.return_value.with_structured_output.assert_called_once_with(TitleGuesses, include_raw=True)
        record = self.records()[-1]
        self.assertEqual(record["event"], "error")
        self.assertEqual(record["error"], "Invalid title schema")
        self.assertEqual(result["results"][0]["status"], "lookup_error")
        self.assertNotIn("raw", record)

    async def test_secrets_are_redacted_from_fields_and_error_strings(self):
        secret = "secret-api-key-for-test"
        with patch.dict(os.environ, {"GOOGLE_API_KEY": secret}), trace_context(self.logger):
            log_event("tool.end", result={"api_key": secret, "error": f"Request failed with key {secret}",
                      "nested": {"authorization": "Bearer hidden", "answer": "hello"}})
        text = self.path.read_text(encoding="utf-8")
        self.assertNotIn(secret, text)
        self.assertNotIn("Bearer hidden", text)
        self.assertEqual(self.records()[0]["result"]["nested"]["answer"], "hello")
        self.assertNotIn(secret, self.logger.readable_path.read_text(encoding="utf-8"))

    async def test_plain_error_result_and_multiple_content_blocks_are_preserved(self):
        self.logger.write("tool.end", result=ToolMessage(content="Tool failed", tool_call_id="a", status="error"))
        self.logger.write("tool.end", result=ToolMessage(content=[
            {"type": "text", "text": '{"rating_source": "predicted", "rating": 2.554}'},
            {"type": "text", "text": "Additional information"}], tool_call_id="b"))
        self.assertEqual(self.records()[0]["result"], "Tool failed")
        self.assertEqual(self.records()[0]["status"], "error")
        self.assertEqual(self.records()[1]["result"], [
            {"rating_source": "predicted", "rating": 2.554}, "Additional information"])


if __name__ == "__main__":
    unittest.main()
