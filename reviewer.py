## Clase de criticos de peliculas


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Reviewer:
    def __init__(self, api_key=OPENAI_API_KEY):
        self.api_key = api_key

    def chat_completion(self, system_prompt, user_prompt):
        # Lógica para interactuar con la API de OpenAI
        
    
    def discuss_movie(self, movie_title):
        # Lógica para discutir la película usando OpenAI API
        pass

    
