from google.generativeai import list_models

models = list_models()
for model in models:
    print(f"Model Name: {model.name}, Display Name: {model.display_name}")
