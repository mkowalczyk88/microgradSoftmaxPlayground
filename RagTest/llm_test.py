# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Locutusque/gpt2-medium-conversational")
response = pipe("<|USER|> What is Tree of Thoughts? <|ASSISTANT|>", max_length = 200, num_return_sequences=1)

print(response)