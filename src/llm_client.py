import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List


class LLMClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-5",
        settings: Optional[Dict[str, object]] = None,
    ) -> str:
        """
        Return the string of the API response using the new OpenAI client.

        Tunable `settings` accepted (forwarded to the API):
          - temperature: float
          - max_completion_tokens: int
          - top_p: float
          - frequency_penalty: float
          - presence_penalty: float
          - n: int
          - stop: List[str] | str | None

        If `settings` is None, sensible defaults are used.
        """


        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # Extract content in a robust way depending on response shape
        try:
            return response.choices[0].message.content
        except Exception:
            try:
                return response.choices[0]["message"]["content"]
            except Exception:
                return str(response)


def extract_json_block(text: str) -> Dict:
    """
    Best-effort helper used by critics/judges to parse STRICT JSON responses.
    - First tries full-string JSON.
    - Then searches the first {...} block and parses that.
    - Returns {} on failure.
    """
    if not text:
        return {}
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find a JSON object block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}
