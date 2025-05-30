from openai import OpenAI
from kaggle_secrets import UserSecretsClient 

openai_key = load_secret('legolas_API')

client = openai.OpenAI(
  api_key=openai_key
)

prompt = "prompt"

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": prompt}
  ]
)
print(completion.choices[0].message.content);
