import os
import chainlit as cl
from openai import OpenAI

# ========================
# CONFIG GERAIS
# ========================
MCP_URL = "https://solutions-garage-ai-gateway-lab.sensedia-eng.com/insurance-mcp/v1/mcp"
MCP_LABEL = "mcp-insurance"

OPENAI_MODEL = "gpt-5.1"#"gpt-5-nano"#"gpt-5.1"  # pode trocar p/ outro modelo compatível
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MCP_TOKEN = os.getenv("MCP_TOKEN")

client = OpenAI(api_key=OPENAI_API_KEY)


def call_llm_with_mcp(user_message: str) -> str:
    """
    Chama o modelo da OpenAI usando o MCP server mcp-insurance como tool.
    Quem fala "protocolo MCP + streamable_http" com teu servidor é a OpenAI,
    não o Chainlit diretamente.
    """

    if not OPENAI_API_KEY:
        return (
            "⚠️ Falta configurar OPENAI_API_KEY no ambiente.\n"
            "Use: export OPENAI_API_KEY='sua_chave_aqui'"
        )

    if not MCP_TOKEN:
        return (
            "⚠️ Falta configurar MCP_TOKEN no ambiente.\n"
            "Use: export MCP_TOKEN='seu_jwt_aqui'"
        )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            tools=[
                {
                    "type": "mcp",
                    "server_label": MCP_LABEL,
                    "server_description": "MCP de seguros da Sensedia",
                    "server_url": MCP_URL,
                    # Token que o MCP precisa para autenticação/autorização
                    "authorization": MCP_TOKEN,
                    # Para demo: deixa o modelo chamar o MCP sem pedir aprovação
                    "require_approval": "never",
                }
            ],
            input=user_message,
        )
    except Exception as e:
        return f"❌ Erro ao chamar OpenAI + MCP: {e}"

    # A Responses API já faz as chamadas ao MCP e monta a resposta final.
    try:
        return resp.output_text  # texto pronto para exibir ao usuário
    except Exception:
        return f"📦 Resposta bruta do modelo: {resp}"
    

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "✅ Chat conectado ao LLM + MCP `mcp-insurance`.\n\n"
            "Você pode fazer perguntas de negócio (seguros) e o modelo decide "
            "quando chamar o MCP."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_text = message.content
    reply = call_llm_with_mcp(user_text)
    await cl.Message(content=reply).send()
