import os
from dotenv import load_dotenv
import openai

class ChatGPTCCaller:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    async def call_api(self, system_prompt, user_prompt, model="gpt-5"):
        """
        Return the string of the API response
        """
        response = await openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

        
        