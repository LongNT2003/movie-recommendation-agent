"""Interactive terminal client for the Gemini recommender agent."""

from __future__ import annotations

import asyncio

from recommender.agent import HistoryEntry, MovieAgent


async def main() -> None:
    history: list[HistoryEntry] = []
    async with MovieAgent() as agent:
        print("Movie agent ready. Type /exit to quit.")
        print(f"Chat log: {agent.trace.readable_path}")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() == "/exit":
                break
            if not query:
                continue
            try:
                answer = await agent.chat(query, history)
            except Exception as exc:
                print(f"Error: {exc}")
                continue
            print(f"Agent: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
