import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

os.environ["GOOGLE_API_KEY"] = load_dotenv().get("GOOGLE_API_KEY")

model = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

Intake_Historian = create_agent(model=model, tools=["google_search"], verbose=True)