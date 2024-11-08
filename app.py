import streamlit as st
import os
import dotenv
import uuid

# check if it's linux so it works on Streamlit Cloud
# if os.name == "posix":

#     __import__("pysqlite3")
#     import sys

#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage

from app_methods import stream_llm_response

dotenv.load_dotenv()

MODELS = [
    "gemini/gemini-1.5-flash",
    "mistral/pixtral-12b-2409",
    "openai/gpt-4o-mini",
]
st.set_page_config(
    page_title="AI course content generator",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stToolbarActions.st-emotion-cache-1p1m4ay.e3g6aar0
        {
            display: none !important;
            pointer-events: none;
        }

        img._profileImage_51w34_76
        {
            visibility: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Title ---
st.html(
    """<h2 style="text-align: center;"><i>E-Learning Course Content Creator</i></h2>"""
)

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi, welcome! How can I help you?"},
    ]

# --- LLMs API keys Side Bar ---
# only for development environment, in production it should return None
with st.sidebar:
    st.write("Add API Key for 1 LLM to open chat section")

    default_gemini_api_key = (
        os.getenv("GEMINI_API_KEY") if os.getenv("GEMINI_API_KEY") is not None else ""
    )
    with st.popover("Gemini"):
        gemini_api_key = st.text_input(
            "Paste your Gemini API Key (https://aistudio.google.com/welcome)",
            value=default_gemini_api_key,
            type="password",
            key="gemini_api_key",
        )

    default_openai_api_key = (
        os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
    )
    with st.popover("OpenAI"):
        openai_api_key = st.text_input(
            "Paste your OpenAI API Key (https://platform.openai.com/)",
            value=default_openai_api_key,
            type="password",
            key="openai_api_key",
        )

    default_mistral_api_key = (
        os.getenv("MISTRAL_API_KEY") if os.getenv("MISTRAL_API_KEY") is not None else ""
    )
    with st.popover("Mistral"):
        mistral_api_key = st.text_input(
            "Paste your Mistral API Key (https://console.mistral.ai/)",
            value=default_mistral_api_key,
            type="password",
            key="mistral_api_key",
        )

# --- Main section ---
# Check if the user has added LLM API Key
missing_openai = (
    openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
)
missing_mistral = mistral_api_key == "" or mistral_api_key is None
missing_gemini = gemini_api_key == "" or gemini_api_key is None

if missing_openai and missing_mistral and missing_gemini:
    st.write("#")
    st.warning("Please add your API Key to test prompts...")
else:
    # Sidebar
    with st.sidebar:
        st.divider()
        st.selectbox(
            "Select a Model",
            [
                model
                for model in MODELS
                if ("gemini" in model and not missing_gemini)
                or ("mistral" in model and not missing_mistral)
                or ("openai" in model and not missing_openai)
            ],
            key="model",
        )

        st.button(
            "Clear Chat",
            on_click=lambda: st.session_state.messages.clear(),
            type="primary",
        )

    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "mistral":
        llm_stream = ChatMistralAI(
            api_key=mistral_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "gemini":
        llm_stream = ChatGoogleGenerativeAI(
            api_key=gemini_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            messages = [
                (
                    HumanMessage(content=m["content"])
                    if m["role"] == "user"
                    else AIMessage(content=m["content"])
                )
                for m in st.session_state.messages
            ]

            st.write_stream(stream_llm_response(llm_stream, messages))

with st.sidebar:
    st.divider()
