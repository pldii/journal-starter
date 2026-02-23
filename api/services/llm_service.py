import os
from openai import AsyncOpenAI
from datetime import datetime, UTC

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

async def analyze_journal_entry(entry_id: str, entry_text: str) -> dict:
    response = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Analyze the journal entry and respond with JSON only: {\"sentiment\": \"positive|negative|neutral\", \"summary\": \"2 sentence summary\", \"topics\": [\"topic1\", \"topic2\"]}"
            },
            {"role": "user", "content": entry_text}
        ],
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    import json
    data = json.loads(result)
    
    return {
        "entry_id": entry_id,
        "sentiment": data["sentiment"],
        "summary": data["summary"],
        "topics": data["topics"],
        "created_at": datetime.now(UTC).isoformat()
    }