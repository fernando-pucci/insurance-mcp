import os
import chainlit as cl
from openai import AsyncOpenAI

MCP_URL = "https://solutions-garage-ai-gateway-lab.sensedia-eng.com/insurance-mcp/v1/mcp"
MCP_LABEL = "mcp-insurance"

OPENAI_MODEL = "gpt-5.1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MCP_TOKEN = os.getenv("MCP_TOKEN")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "✅ Chat conectado ao LLM + MCP `mcp-insurance`.\n\n"
            "Você pode fazer perguntas de negócio (seguros) e o modelo decide quando chamar o MCP."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    if not OPENAI_API_KEY:
        await cl.Message(content="⚠️ Falta configurar OPENAI_API_KEY no ambiente.").send()
        return

    if not MCP_TOKEN:
        await cl.Message(content="⚠️ Falta configurar MCP_TOKEN no ambiente.").send()
        return

    # Mensagem “vazia” para ir atualizando token a token
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Streaming da Responses API
        stream = await client.responses.create(
            model=OPENAI_MODEL,
            tools=[
                {
                    "type": "mcp",
                    "server_label": MCP_LABEL,
                    "server_description": "MCP de seguros da Sensedia",
                    "server_url": MCP_URL,
                    "authorization": MCP_TOKEN,
                    "require_approval": "never",
                }
            ],
            input=message.content,
            stream=True,
        )

        async for event in stream:
            # Eventos de texto incremental
            if getattr(event, "type", None) == "response.output_text.delta":
                await msg.stream_token(event.delta)

        await msg.update()

    except Exception as e:
        await msg.update()
        await cl.Message(content=f"❌ Erro ao chamar OpenAI + MCP (stream): {e}").send()
