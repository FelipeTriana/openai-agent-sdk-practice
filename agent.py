import asyncio
import os

from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Carga las variables del archivo .env en os.environ.
load_dotenv()

# Configura el cliente HTTP apuntando al endpoint local de Ollama,
# que expone una API REST compatible con la de OpenAI.
client = AsyncOpenAI(
    base_url=os.environ["OLLAMA_BASE_URL"],
    api_key=os.environ["OLLAMA_API_KEY"],
)

# Redirige todas las llamadas del SDK a ese cliente local.
set_default_openai_client(client)

# Desactiva el envío de trazas a los servidores de OpenAI.
set_tracing_disabled(True)

# Define el agente.
# "instructions" es el system prompt enviado al LLM en cada invocación.
agent = Agent(
    name="openai-agent-sdk-practice",
    instructions="Eres un comediante. Cuando alguien te hable, cuenta un chiste gracioso y original.",
    model=os.environ["OLLAMA_MODEL"],
)


async def main() -> None:
    result = await Runner.run(agent, input="Hola, cuéntame un chiste.")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
