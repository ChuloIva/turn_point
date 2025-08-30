import ollama

# Pull a model (first time only)
# ollama pull llama3.2

# Simple text generation
response = ollama.generate(model='llama3.2', prompt='Why is the sky blue?')
print(response['response'])

# Chat interface
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Explain quantum computing'}
])
print(response['message']['content'])

# Async support
import asyncio
from ollama import AsyncClient

async def chat():
    message = {'role': 'user', 'content': 'Why is the sky blue?'}
    response = await AsyncClient().chat(model='llama3.2', messages=[message])
    print(response['message']['content'])

asyncio.run(chat())