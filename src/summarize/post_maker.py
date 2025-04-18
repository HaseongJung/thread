from .gemini_client import GeminiClient
from src.summarize.prompt import PromptManager
from src.summarize.gemini_client import GeminiClient

class PostMaker:
    def __init__(self, model_name: str = "gemma-3-12b-it"):
        self.client = GeminiClient(model_name=model_name)
        self.prompt_manager = PromptManager()

    
    def generate_post(self, meata_data_path: str) -> str:
        """
        Generates a post using the Gemini client.
        """

        # Prepare the data for the prompt
        meta_data = self.prompt_manager.prepare_data(meata_data_path)
        prompt = self.prompt_manager.create_prompt(meta_data)

        # Generate the post using the Gemini client
        response = self.client.generate_text(prompt)
        return response
    

    def save_post(self, post: str, file_path: str) -> None:
        """
        Saves the generated post to a file.
        """
        with open(file_path, 'w') as file:
            file.write(post)
        print(f"Post saved to {file_path}")