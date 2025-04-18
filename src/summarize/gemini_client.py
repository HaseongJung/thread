import os
import google.generativeai as genai

class GeminiClient:
    def __init__(self, model_name: str="gemini-2.5-pro-exp-03-25"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set the GENAI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_text(self, prompt: str) -> str:
        """
        Generate content using the Gemini model.
        
        Args:
            prompt (str): The input prompt for content generation.
        
        Returns:
            str: The generated content.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Error generating content: {e}")