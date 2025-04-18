from src.summarize.prompt import PromptManager
from src.summarize.gemini_client import GeminiClient

# prepare the prompt
prompt_manager = PromptManager()
data_path = "output/topic_modeling/20250418_0600/Documents/Topic-1_20250416_1251_국민_대통령_국회_경선_대선.csv"
meta_data = prompt_manager.prepare_data(data_path)
prompt = prompt_manager.create_prompt(meta_data=meta_data)
# print(prompt)

# get the post
gemini = GeminiClient()
response = gemini.generate_text(prompt)
print(response)