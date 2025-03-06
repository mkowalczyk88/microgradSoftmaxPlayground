# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Locutusque/gpt2-medium-conversational")
response = pipe("<|USER|> What is the meaning of life? <|ASSISTANT|>", max_length = 100, num_return_sequences=3)

print(response)