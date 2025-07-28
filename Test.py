import google.generativeai as genai

# Set up your API key here
genai.configure(api_key="YOUR_API_KEY")

# List available models
models = genai.list_models()
for model in models:
    print(f"Model Name: {model.name}")

