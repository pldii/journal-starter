import json
import os
from datetime import UTC, datetime

from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

async def analyze_journal_entry(entry_id: str, entry_text: str) -> dict:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    system_prompt = (
        "You are a journal analysis assistant. "
        "Analyze the provided journal entry and return a JSON object with these fields: "
        "sentiment (must be 'positive', 'negative', or 'neutral'), "
        "summary (exactly 2 sentences describing the entry), "
        "and topics (a list of 2 to 4 key topics as strings)."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": entry_text},
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM")

    data = json.loads(content)

    return {
        "entry_id": entry_id,
        "sentiment": data["sentiment"],
        "summary": data["summary"],
        "topics": data["topics"],
        "created_at": datetime.now(UTC).isoformat(),
    }