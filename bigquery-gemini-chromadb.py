# Add this at the very top of your file, before any other imports
import os
import sys
import streamlit as st
import json
import tempfile
import time
import plotly.express as px
import plotly.graph_objects as fig

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
from dotenv import load_dotenv
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery

# Load environment variables from .env file
load_dotenv()

# Try to import credentials, but don't fail if not available
try:
    from credentials import get_gemini_api_key, get_google_credentials_json
    has_credentials_module = True
except ImportError:
    has_credentials_module = False

GCP_PROJECT_ID = 'hitech-dados'
BQ_DATASET = 'seat'
GCP_REGION = "us-central1"

# Set up the Streamlit page
st.set_page_config(page_title="Consulta em Linguagem Natural - Impulso", layout="wide")

# Initialize Vanna with caching
@st.cache_resource(ttl=3600)
def setup_vanna():
    # Try to get credentials from various sources
    credentials = None
    gemini_api_key = None
    
    # First try Streamlit secrets
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
        try:
            credentials_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            st.sidebar.success("✅ Connected to BigQuery with Streamlit secrets")
        except Exception as e:
            st.sidebar.error(f"Error with Streamlit secrets: {e}")
    
    # Then try credentials module
    elif has_credentials_module:
        try:
            google_credentials_json = get_google_credentials_json()
            credentials_info = json.loads(google_credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            st.sidebar.success("✅ Connected to BigQuery with local credentials")
        except Exception as e:
            st.sidebar.warning(f"Could not use local credentials: {e}")
    
    # Finally try environment variable
    elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        try:
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            st.sidebar.success("✅ Connected to BigQuery with environment credentials")
        except Exception as e:
            st.sidebar.warning(f"Could not use environment credentials: {e}")
    
    # Get Gemini API key from various sources
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    elif has_credentials_module:
        try:
            gemini_api_key = get_gemini_api_key()
        except Exception:
            pass
    
    if not gemini_api_key:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        st.sidebar.error("❌ No Gemini API key found")
    else:
        st.sidebar.success("✅ Gemini API key configured")
    
    # Create BigQuery client if we have credentials
    client = None
    if credentials:
        client = bigquery.Client(credentials=credentials, project=GCP_PROJECT_ID)
    
    # Initialize Vanna
    class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            GoogleGeminiChat.__init__(self, config={'api_key': gemini_api_key, 'model': 'gemini-pro'})
    
    vn = MyVanna()
    
    try:
        # Connect to BigQuery using the client we created
        if client:
            vn.bigquery_client = client
            vn.bigquery_project_id = GCP_PROJECT_ID
        else:
            # Try default connection method as fallback
            vn.connect_to_bigquery(project_id=GCP_PROJECT_ID)
        
        # Training data remains the same
        vn.train(ddl=f"""
        CREATE TABLE `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` (
          id bigserial NOT NULL,
            resource_id int8 NULL,
            project_id int8 NULL,
            start_at date NOT NULL,
            end_at date NOT NULL,
            created_at timestamp NOT NULL,
            updated_at timestamp NOT NULL,
            billing_rate numeric(14, 2) NULL,
            resource_rate numeric(14, 2) NULL,
            client_id int8 NOT NULL,
            opportunity_id int8 NULL,
            xp_manager_id int8 NULL,
            "uuid" uuid NULL,
            replacement_id int8 NULL,
            modification_type int4 NULL,
            modification_reason int4 NULL,
            unarchive_history jsonb DEFAULT '[]'::jsonb NULL
        );
        """)
        
        vn.train(documentation="A tabela 'allocations' contém informações sobre os impusers alocados. Quando a alocação iniciou e terminou, em qual projetos estão alocados. Valor hora que o impulser recebe e o valor que a empresa recebe por esse impulser. Qual o cliente o impulser está alocado. Por qual oportunidade de emprego ele foi aprovado e alocado. e qual funcionário dez o acompanhamento.")
        
        vn.train(question="Quantos impulsers temos alocados hoje?",
                sql=f"select count(*) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` where end_at >= current_date")
        
        vn.train(question="Quantos impulsers iniciaram suas alocações em abril de 2025?",
                sql=f"select count(*) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` where start_at >= '2025-04-01'")
        
        vn.train(question="Qual é a porcentagem de fee por impulser que foram alocados a partir de em abril de 2025?",
                sql=f"select round((1 - a.resource_rate / (a.billing_rate * (1-0.1656)))*100,2) as percent_fee FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` a where start_at >= '2025-04-01'")
        
        vn.train(question="Quantos projetos receberam impulsers alocados no mes de abril de 2025?",
                sql=f"select distinct count(project_id) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` where start_at >= '2025-04-01'")
        
    except Exception as e:
        st.error(f"Erro ao conectar ao BigQuery: {e}")
    
    return vn

# Cached functions similar to vanna_calls.py
@st.cache_data(ttl=3600)
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()

@st.cache_data(show_spinner="Gerando consulta SQL...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question)

@st.cache_data(show_spinner="Executando consulta...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Gerando código para visualização...")
def generate_plotly_code_cached(question: str, sql: str, df):
    vn = setup_vanna()
    return vn.generate_plotly_code(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Gerando visualização...")
def generate_plot_cached(code: str, df):
    try:
        # Create a local namespace for execution
        local_vars = {"px": px, "fig": fig, "df": df}
        exec(code, globals(), local_vars)
        return local_vars.get("fig")
    except Exception as e:
        st.error(f"Error generating plot: {e}")
        return None

@st.cache_data(show_spinner="Gerando perguntas relacionadas...")
def generate_followup_cached(question: str, sql: str, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Verificando se devo gerar gráfico...")
def should_generate_chart_cached(question: str, sql: str, df):
    vn = setup_vanna()
    return vn.should_generate_chart(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Verificando SQL...")
def is_sql_valid_cached(sql: str):
    # Simple validation - could be enhanced
    return isinstance(sql, str) and "select" in sql.lower()

@st.cache_data(show_spinner="Gerando resumo...")
def generate_summary_cached(question: str, df):
    vn = setup_vanna()
    return vn.generate_answer(question=question, df=df)

# UI Setup
avatar_url = "https://vanna.ai/img/vanna.svg"

st.title("Consulta em Linguagem Natural - Impulso")
st.write("Faça perguntas sobre os dados de alocação de impulsers em linguagem natural")

# Sidebar configuration
st.sidebar.title("Configurações")
st.sidebar.checkbox("Mostrar SQL", value=True, key="show_sql")
st.sidebar.checkbox("Mostrar Tabela", value=True, key="show_table")
st.sidebar.checkbox("Mostrar Código Plotly", value=False, key="show_plotly_code")
st.sidebar.checkbox("Mostrar Gráfico", value=True, key="show_chart")
st.sidebar.checkbox("Mostrar Resumo", value=True, key="show_summary")
st.sidebar.checkbox("Mostrar Perguntas Relacionadas", value=True, key="show_followup")
st.sidebar.button("Resetar", on_click=lambda: st.session_state.pop("my_question", None), use_container_width=True)

# Display training data if requested
if st.sidebar.checkbox("Mostrar dados de treinamento"):
    vn = setup_vanna()
    training_data = vn.get_training_data()
    st.sidebar.json(training_data)

def set_question(question):
    st.session_state["my_question"] = question

# Suggested questions
assistant_message_suggested = st.chat_message("assistant", avatar=avatar_url)
if assistant_message_suggested.button("Clique para mostrar perguntas sugeridas"):
    st.session_state["my_question"] = None
    questions = generate_questions_cached()
    for question in questions:
        time.sleep(0.05)
        st.button(question, on_click=set_question, args=(question,))

# Get question from session state or input
my_question = st.session_state.get("my_question", None)
if my_question is None:
    my_question = st.chat_input("Faça uma pergunta sobre seus dados")

# Process the question
if my_question:
    st.session_state["my_question"] = my_question
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    sql = generate_sql_cached(question=my_question)

    if sql:
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message("assistant", avatar=avatar_url)
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
        else:
            assistant_message = st.chat_message("assistant", avatar=avatar_url)
            assistant_message.write(sql)
            st.stop()

        df = run_sql_cached(sql=sql)

        if df is not None:
            st.session_state["df"] = df

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message("assistant", avatar=avatar_url)
                if len(df) > 10:
                    assistant_message_table.text("Primeiras 10 linhas dos dados")
                    assistant_message_table.dataframe(df.head(10))
                else:
                    assistant_message_table.dataframe(df)

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):
                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    assistant_message_plotly_code = st.chat_message("assistant", avatar=avatar_url)
                    assistant_message_plotly_code.code(code, language="python", line_numbers=True)

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        assistant_message_chart = st.chat_message("assistant", avatar=avatar_url)
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("Não foi possível gerar um gráfico")

            if st.session_state.get("show_summary", True):
                assistant_message_summary = st.chat_message("assistant", avatar=avatar_url)
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    assistant_message_summary.text(summary)

            if st.session_state.get("show_followup", True):
                assistant_message_followup = st.chat_message("assistant", avatar=avatar_url)
                followup_questions = generate_followup_cached(question=my_question, sql=sql, df=df)
                st.session_state["df"] = None

                if len(followup_questions) > 0:
                    assistant_message_followup.text("Aqui estão algumas perguntas relacionadas")
                    # Print the first 5 follow-up questions
                    for question in followup_questions[:5]:
                        assistant_message_followup.button(question, on_click=set_question, args=(question,))

    else:
        assistant_message_error = st.chat_message("assistant", avatar=avatar_url)
        assistant_message_error.error("Não foi possível gerar SQL para essa pergunta")