import yaml
from pathlib import Path
import pandas as pd

class PromptManager:
    def __init__(self):
        self.template_path = Path(__file__).parent.parent.parent / "config" / "prompts" / "thread_template.yaml"
        self.template = self.load_template()

    def load_template(self) -> str:
        """
        Load the prompt template from a YAML file.
        """
        with open(self.template_path, 'r') as file:
            template = yaml.safe_load(file)
        return template["prompt"]
        
    def prepare_data(self, df_path: str) -> str:
        """
        Prepare the data for the prompt.
        """
        # Convert the DataFrame to a string format
        df = pd.read_csv(df_path)
        # 각 행을 지정된 형식의 문자열로 변환
        formatted_articles = df.apply(
            lambda row: f"Title: {row['title']}, Description: {row['description']}\n",
            axis=1
        )
        # 모든 문자열을 하나로 결합
        meta_data = ''.join(formatted_articles)
        return meta_data

    def create_prompt(self, meta_data: str) -> str:
        """
        Create the prompt by loading the template and preparing the data.
        """
        
        prompt = f'{self.template}\n\n\n\n{meta_data}'

        return prompt