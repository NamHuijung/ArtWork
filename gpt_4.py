import os
import openai
from openai import OpenAI
    
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a helpful math tuton. Help me with my math homework!"},
    {"role": "user", "content": "Hello! Could you solve 2+2?"}  
  ]
)

print("Assistant: " + completion.choices[0].message.content)