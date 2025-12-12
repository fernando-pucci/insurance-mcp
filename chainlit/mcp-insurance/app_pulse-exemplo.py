import chainlit as cl
import os
import sys
import traceback as tb
from chainlit.mcp import McpConnection
from langchain_core.messages import (
    AIMessageChunk,
)
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from mcp import ClientSession
from src.utils.bedrock import get_bedrock_client, get_chat_model
from src.utils.models import InferenceConfig, ModelId, ThinkingConfig
from typing import cast

import pandas as pd
from sqlalchemy import create_engine, text


# Recupera a URL do banco do próprio Chainlit
db_url = os.environ.get("DATABASE_URL")
engine = create_engine(db_url)


# Função para buscar o histórico diretamente no banco
async def get_history(thread_id):
    query = text("""
        SELECT output, "createdAt"
        FROM "Step"
        WHERE "threadId" = :thread_id
        AND   ("name" = 'tools') or ("name" = 'User' and output like '%@context%')
        ORDER BY "createdAt" ASC
    """)

    with engine.connect() as conn:
        results = conn.execute(query, {"thread_id": thread_id})
        messages = results.fetchall()

    return messages

logger.remove()
logger.add(sys.stderr, level=os.getenv('LOG_LEVEL', 'ERROR'))

bedrock_client = get_bedrock_client()
chat_model = get_chat_model(
    model_id=ModelId.ANTHROPIC_CLAUDE_3_7_SONNET,
    inference_config=InferenceConfig(temperature=1, max_tokens=4096 * 8),
    thinking_config=ThinkingConfig(budget_tokens=1024),
    client=bedrock_client,
)


@cl.on_mcp_connect  # type: ignore
async def on_mcp(connection: McpConnection, session: ClientSession) -> None:
    """Called when an MCP connection is established."""
    await session.initialize()
    tools = await load_mcp_tools(session)
    agent = create_react_agent(
        chat_model,
        tools,
        prompt="You are a helpful assistant. You must use the tools provided to you to answer the user's question.",
    )
    agent.recursion_limit = 100

    cl.user_session.set('agent', agent)
    cl.user_session.set('mcp_session', session)
    cl.user_session.set('mcp_tools', tools)


@cl.on_mcp_disconnect  # type: ignore
async def on_mcp_disconnect(name: str, session: ClientSession) -> None:
    """Called when an MCP connection is terminated."""
    if isinstance(cl.user_session.get('mcp_session'), ClientSession):
        await session.__aexit__(None, None, None)
        cl.user_session.set('mcp_session', None)
        cl.user_session.set('mcp_name', None)
        cl.user_session.set('mcp_tools', {})
        logger.debug(f'Disconnected from MCP server: {name}')


@cl.on_chat_start
async def on_chat_start():
    # Carregar o CSV com o conjunto de regras
    df = pd.read_csv("rule_set.csv")

    # Armazenar o conteúdo no contexto da sessão
    cl.user_session.set("rule_set", df.to_dict(orient="records"))

    #await cl.Message(
    #    content="Conjunto de boas práticas das APIs da Sensedia carregado no contexto. \nPara referenciar as regras no chat, basta que seu mensagem inclua o termo **rule_set.csv.**\nPara considerar o histórico do chat, utilize o termo **chat_history**."
    #).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_text_message = message.content.lower()
    base_prompt = f"Usuário perguntou: '{message.content}'\n"

    # Recupera as regras do contexto
    rule_Set = cl.user_session.get("rule_set", [])

    # Verifica se o usuário mencionou 'rule_set.csv'
    if "@rule_set.csv" in user_text_message:
        # Resumo das regras (personalize conforme necessidade)
        base_prompt += f"\nO usuário mencionou o arquivo rule_set.csv. Aqui estão as boas práticas relevantes:\n{rule_Set}\n"
    

    # Verifica se o usuário mencionou 'chat_history'
    if "@chat_history" in user_text_message:
        thread_id = message.thread_id
        chat_history = await get_history(thread_id)
        if not chat_history:
            await cl.Message(content="Não há histórico salvo no banco ainda.").send()
            return
        else:
            formatted_history = "\n".join(f"[{dt}] {msg}" for msg, dt in chat_history)
            base_prompt += f"\nConsidere o histórico de respostas anteriores dos MCPs para evitar novas chamadas desnecessárias:\n{formatted_history}\n"


    """Process user messages and generate responses using the Bedrock model."""
    config = RunnableConfig(
        configurable={'thread_id': cl.context.session.id},
        recursion_limit=100
    )
    agent = cast(CompiledStateGraph, cl.user_session.get('agent'))
    if not agent:
        await cl.Message(content='Error: Chat model not initialized.').send()
        return

    cb = cl.AsyncLangchainCallbackHandler()

    try:
        # Create a message for streaming
        response_message = cl.Message(content='')

        # Stream the response using the LangChain callback handler
        # Update the config to include callbacks
        config['callbacks'] = [cb]
        async for msg, metadata in agent.astream(
            #{'messages': message.content},
            {'messages': base_prompt},
            stream_mode='messages',
            config=config,
        ):
            # Handle AIMessageChunks with text content for streaming
            if isinstance(msg, AIMessageChunk) and msg.content:
                # If content is a string, stream it directly
                if isinstance(msg.content, str):
                    await response_message.stream_token(msg.content)
                # If content is a list with dictionaries that have text
                elif (
                    isinstance(msg.content, list)
                    and len(msg.content) > 0
                    and isinstance(msg.content[0], dict)
                    and msg.content[0].get('type') == 'text'
                    and 'text' in msg.content[0]
                ):
                    await response_message.stream_token(msg.content[0]['text'])

        # Send the complete message
        await response_message.send()

    except Exception as e:
        # Error handling
        err_msg = cl.Message(content=f'Error: {str(e)}')
        await err_msg.send()
        logger.error(tb.format_exc())


from typing import Dict, Optional
import chainlit as cl

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  if provider_id == "google":
    if raw_user_data["hd"] == "sensedia.com":
      return default_user
  return None
