"""
Chainlit + OpenAI (GPT-5) + MCP (streamable-http) — app.py (MELHORADO)

Melhorias vs versão anterior:
- NÃO exige CPF+placa. Exige apenas 1 identificador (CPF OU placa OU apólice OU e-mail OU telefone).
- Se tiver CPF e faltar placa, o modelo deve chamar MCP Lista_clientes para obter placa (sem perguntar de novo).
- Mantém estado (cpf/placa/identificador) e histórico curto.
- Em SINISTRO: tool_choice="required" para sempre tentar MCP.
- Streaming na UI (digitando).
- /reset para limpar contexto.

Requisitos:
  pip install -U chainlit openai

Env:
  export OPENAI_API_KEY="..."
  export MCP_TOKEN="Bearer ...."  (ou como seu gateway exige)
Opcional:
  export MCP_URL="https://..."
  export MCP_LABEL="mcp-insurance"
  export OPENAI_MODEL="gpt-5.1"
"""

import os
import re
from typing import List, Dict, Any, Optional

import chainlit as cl
from openai import AsyncOpenAI

# =========================
# Config
# =========================

MCP_URL = os.getenv(
    "MCP_URL",
    "https://solutions-garage-ai-gateway-lab.sensedia-eng.com/insurance-mcp/v1/mcp",
)
MCP_LABEL = os.getenv("MCP_LABEL", "mcp-insurance")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MCP_TOKEN = os.getenv("MCP_TOKEN")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =========================
# MCP tools (nomes exatos)
# =========================

TOOL_ELIGIBILIDADE = "[Seguradora-MCP---DEMO]-API-Seguradora_Verifica_se_o_cliente_ou_apolice_e_elegivel_para_um_dete"
TOOL_LISTA_SINISTROS = "[Seguradora-MCP---DEMO]-API-Seguradora_Lista_sinistros_claims_com_dados_de_cliente_apolice_stat"
TOOL_LISTA_VEICULOS_LOCADORAS = "[Seguradora-MCP---DEMO]-API-Seguradora_Lista_veiculos_disponiveis_das_locadoras_parceiras"
TOOL_DADOS_APOLICES = "[Seguradora-MCP---DEMO]-API-Seguradora_Dados_gerais_de_apolices_de_seguro"
TOOL_LISTA_OFICINAS = "[Seguradora-MCP---DEMO]-API-Seguradora_Lista_oficinas_credenciadas_trazendo_dados_comor_cidade_"
TOOL_LISTA_CLIENTES = "[Seguradora-MCP---DEMO]-API-Seguradora_Lista_clientes_e_retorna_dados_como_nome_documento_telef"
TOOL_OFERTAS_CARRO_RESERVA = "[Seguradora-MCP---DEMO]-API-Seguradora_Retorna_ofertas_de_veiculos_de_locadoras_parceiras_para_"

SINISTRO_ALLOWED_TOOLS = [
    TOOL_LISTA_CLIENTES,          # chave para resolver placa/apólice a partir de CPF etc.
    TOOL_DADOS_APOLICES,
    TOOL_ELIGIBILIDADE,
    TOOL_LISTA_SINISTROS,
    TOOL_LISTA_OFICINAS,
    TOOL_OFERTAS_CARRO_RESERVA,
    TOOL_LISTA_VEICULOS_LOCADORAS,
]

GERAL_ALLOWED_TOOLS = [
    TOOL_LISTA_CLIENTES,
    TOOL_DADOS_APOLICES,
    TOOL_LISTA_SINISTROS,
]

# =========================
# Intent + parsers
# =========================

SINISTRO_KEYWORDS = [
    "bati", "batida", "colisão", "colisao", "acidente", "capotei",
    "roubo", "furt", "assalto", "pane", "guincho", "carro reserva",
    "sinistro", "claim", "quebrou", "perda total"
]

PLATE_RE = re.compile(r"\b[A-Z]{3}[0-9][A-Z0-9][0-9]{2}\b", re.IGNORECASE)  # Mercosul/antigo
CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}\-?\d{2}\b")
CNPJ_RE = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}\-?\d{2}\b")
EMAIL_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}\-?\d{4}\b")

def is_sinistro_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in SINISTRO_KEYWORDS)

def extract_cpf(text: str) -> Optional[str]:
    if not text:
        return None
    m = CPF_RE.search(text)
    return re.sub(r"\D", "", m.group(0)) if m else None

def extract_cnpj(text: str) -> Optional[str]:
    if not text:
        return None
    m = CNPJ_RE.search(text)
    return re.sub(r"\D", "", m.group(0)) if m else None

def extract_plate(text: str) -> Optional[str]:
    if not text:
        return None
    m = PLATE_RE.search(text.upper())
    return m.group(0).upper() if m else None

def extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None

def extract_phone(text: str) -> Optional[str]:
    if not text:
        return None
    m = PHONE_RE.search(text)
    return m.group(0) if m else None

def detect_event(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "colis" in t or "bati" in t or "acidente" in t:
        return "colisão"
    if "roubo" in t or "furto" in t or "assalto" in t:
        return "roubo/furto"
    if "pane" in t or "quebrou" in t:
        return "pane"
    return None

# =========================
# Session state + history
# =========================

STATE_DEFAULT = {
    "cpf": None,
    "cnpj": None,
    "placa": None,
    "email": None,
    "telefone": None,
    "evento": None,
}

def get_state() -> Dict[str, Any]:
    state = cl.user_session.get("state")
    if not state:
        state = dict(STATE_DEFAULT)
        cl.user_session.set("state", state)
    return state

def set_state(**kwargs):
    state = get_state()
    for k, v in kwargs.items():
        if v:
            state[k] = v
    cl.user_session.set("state", state)

def clear_state():
    cl.user_session.set("state", dict(STATE_DEFAULT))
    cl.user_session.set("history", [])

def get_history() -> List[Dict[str, str]]:
    return cl.user_session.get("history", [])

def append_history(role: str, content: str, max_items: int = 12):
    hist = get_history()
    hist.append({"role": role, "content": content})
    cl.user_session.set("history", hist[-max_items:])

def has_any_identifier(state: Dict[str, Any]) -> bool:
    return any([
        state.get("cpf"),
        state.get("cnpj"),
        state.get("placa"),
        state.get("email"),
        state.get("telefone"),
    ])

# =========================
# MCP block
# =========================

def build_mcp_tool_block(allowed_tools: List[str]) -> Dict[str, Any]:
    return {
        "type": "mcp",
        "server_label": MCP_LABEL,
        "server_description": "MCP de Seguradora (apólices, sinistros, elegibilidade, oficinas, carro reserva).",
        "server_url": MCP_URL,
        "authorization": MCP_TOKEN,
        "require_approval": "never",
        "allowed_tools": allowed_tools,
    }

# =========================
# Prompting
# =========================

def base_instructions() -> str:
    return (
        "Você é um assistente de sinistros de seguradora.\n"
        "Objetivo: orientar o cliente usando DADOS REAIS via MCP.\n\n"
        "Regras críticas:\n"
        "1) NÃO invente cobertura/benefícios. Para cobertura/franquia/guincho/carro reserva/oficina, consulte o MCP.\n"
        "2) Se houver QUALQUER identificador (CPF/CNPJ OU placa OU e-mail OU telefone), NÃO peça outro identificador: "
        "use o MCP para localizar cliente/apólice e obter o que falta.\n"
        "   - Exemplo: se tiver CPF e faltar placa, chame a tool de LISTA_CLIENTES para obter placa.\n"
        "3) Só pergunte identificador se NÃO houver nenhum no contexto.\n"
        "4) Se o contexto já tiver CPF/placa, NÃO repita a pergunta.\n"
        "5) Responda em PT-BR, com passos claros e objetivos.\n"
    )

def sinistro_instructions() -> str:
    return (
        base_instructions()
        + "\nCenário: SINISTRO.\n"
          "Você DEVE consultar o MCP antes de orientar.\n"
          "Fluxo recomendado:\n"
          "- Use LISTA_CLIENTES para localizar cliente por CPF/CNPJ/placa/e-mail/telefone e obter dados faltantes.\n"
          "- Use DADOS_APOLICES para obter dados da apólice.\n"
          "- Use VERIFICA_ELEGIBILIDADE para confirmar guincho/carro reserva/etc conforme evento.\n"
          "- Se útil, liste oficinas credenciadas e ofertas de carro reserva.\n"
    )

def geral_instructions() -> str:
    return base_instructions() + "\nCenário: GERAL.\n"

def build_state_context(state: Dict[str, Any]) -> str:
    return (
        "CONTEXTO DA CONVERSA (estado confirmado pelo usuário):\n"
        f"- CPF: {state.get('cpf') or 'NÃO INFORMADO'}\n"
        f"- CNPJ: {state.get('cnpj') or 'NÃO INFORMADO'}\n"
        f"- Placa: {state.get('placa') or 'NÃO INFORMADO'}\n"
        f"- E-mail: {state.get('email') or 'NÃO INFORMADO'}\n"
        f"- Telefone: {state.get('telefone') or 'NÃO INFORMADO'}\n"
        f"- Evento: {state.get('evento') or 'NÃO INFORMADO'}\n\n"
        "Regra: Se QUALQUER identificador estiver informado acima, use o MCP para buscar os demais dados. "
        "Não peça placa se já há CPF, por exemplo.\n"
    )

def prompt_for_one_identifier() -> str:
    return (
        "Para eu consultar sua apólice no MCP e te orientar com base na cobertura, me envie **apenas 1** destes dados:\n"
        "- **CPF/CNPJ**, ou\n"
        "- **placa**, ou\n"
        "- **e-mail/telefone** cadastrado.\n\n"
        "Ex.: `CPF 123.456.789-10` ou `placa ABC1D23`."
    )

# =========================
# Chainlit hooks
# =========================

@cl.on_chat_start
async def on_chat_start():
    if not OPENAI_API_KEY:
        await cl.Message(content="⚠️ Configure OPENAI_API_KEY no ambiente.").send()
        return
    if not MCP_TOKEN:
        await cl.Message(content="⚠️ Configure MCP_TOKEN no ambiente (ex.: `Bearer ...`).").send()
        return

    clear_state()

    await cl.Message(
        content=(
            "✅ Pronto! Conectado ao GPT-5 e ao MCP da Seguradora.\n"
            "Dica: em sinistro, basta você informar **CPF** ou **placa** (um deles já resolve).\n"
            "Comando: **/reset** limpa o contexto."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    if not OPENAI_API_KEY or not MCP_TOKEN:
        await cl.Message(content="⚠️ Faltam variáveis de ambiente (OPENAI_API_KEY / MCP_TOKEN).").send()
        return

    user_text = (message.content or "").strip()

    if user_text.lower() in ["/reset", "reset", "/restart"]:
        clear_state()
        await cl.Message(content="🔄 Contexto limpo. O que aconteceu?").send()
        return

    # Atualiza estado com o que o usuário trouxe
    set_state(
        cpf=extract_cpf(user_text),
        cnpj=extract_cnpj(user_text),
        placa=extract_plate(user_text),
        email=extract_email(user_text),
        telefone=extract_phone(user_text),
        evento=detect_event(user_text),
    )
    state = get_state()

    append_history("user", user_text)

    # Intent
    sinistro = is_sinistro_intent(user_text) or (state.get("evento") is not None)
    allowed_tools = SINISTRO_ALLOWED_TOOLS if sinistro else GERAL_ALLOWED_TOOLS
    instructions = sinistro_instructions() if sinistro else geral_instructions()
    tool_choice = "required" if sinistro else "auto"

    # Se é sinistro e não há NENHUM identificador, pede só 1 e para.
    if sinistro and not has_any_identifier(state):
        await cl.Message(content=prompt_for_one_identifier()).send()
        return

    # Streaming output
    out = cl.Message(content="")
    await out.send()

    step = cl.Step(name="MCP/LLM", type="tool")
    await step.__aenter__()
    step.output = "Consultando sistemas (MCP) e montando orientação…"
    await step.update()

    try:
        # Input = contexto + histórico curto
        state_context = build_state_context(state)
        hist = get_history()
        input_messages = [{"role": "system", "content": state_context}, *hist]

        stream = await client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            tools=[build_mcp_tool_block(allowed_tools)],
            tool_choice=tool_choice,
            max_tool_calls=4 if sinistro else 2,
            input=input_messages,
            stream=True,
        )

        assistant_text_accum: List[str] = []

        async for event in stream:
            et = getattr(event, "type", "") or ""

            if et == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    assistant_text_accum.append(delta)
                    await out.stream_token(delta)

            # feedback visual quando houver eventos de tool/mcp
            if "tool" in et or "mcp" in et:
                step.output = f"Executando chamadas via MCP… ({et})"
                await step.update()

        await out.update()
        step.output = "Concluído."
        await step.update()

        full_assistant_text = "".join(assistant_text_accum).strip()
        if full_assistant_text:
            append_history("assistant", full_assistant_text)

    except Exception as e:
        await out.update()
        step.output = f"Erro: {e}"
        await step.update()
        await cl.Message(content=f"❌ Erro ao chamar GPT/MCP: {e}").send()
    finally:
        await step.__aexit__(None, None, None)
