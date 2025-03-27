import os
from huggingface_hub import InferenceClient

# Load API token from environment variable
api_key = os.getenv("HF_API_TOKEN")

# Initialize the client
client = InferenceClient(
    model="Qwen/Qwen2-1.5B-Instruct",
    api_key=api_key
)

# Define the prompt
prompt = "Explain reinforcement learning in simple terms."

# Generate response
output = client.text_generation(prompt, max_new_tokens=512)

# Print the response
print(output)
