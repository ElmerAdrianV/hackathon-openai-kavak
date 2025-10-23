import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, Dict, List


class ChatGPTCCaller:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def call_api(
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
        default_settings = {
            "max_completion_tokens": 512,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "n": 1,
        }

        call_kwargs = default_settings.copy()
        if settings:
            call_kwargs.update(settings)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **call_kwargs,
        )

        # Extract content in a robust way depending on response shape
        try:
            return response.choices[0].message.content
        except Exception:
            try:
                return response.choices[0]["message"]["content"]
            except Exception:
                return str(response)


if __name__ == "__main__":
    caller = ChatGPTCCaller()

    system_prompt = "You are a helpful assistant."
    user_prompt = "Hello, how can you assist me today?"
    response = caller.call_api(system_prompt, user_prompt, model="gpt-5")
    print(response)