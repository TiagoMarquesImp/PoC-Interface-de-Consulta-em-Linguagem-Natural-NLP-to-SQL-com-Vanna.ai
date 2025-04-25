# Add this at the very top of your file, before any other imports
import os
import sys
import streamlit as st

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

GCP_PROJECT_ID = 'hitech-dados'
BQ_DATASET = 'seat'
GCP_REGION = "us-central1"

# Set up the Streamlit page
st.set_page_config(page_title="Consulta em Linguagem Natural - Impulso", layout="wide")
st.title("Consulta em Linguagem Natural para BigQuery")
st.write("Faça perguntas sobre os dados de alocação de impulsers em linguagem natural")

# Initialize Vanna with caching
@st.cache_resource(ttl=3600)
def setup_vanna():
    class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            GoogleGeminiChat.__init__(self, config={'api_key': os.getenv('GEMINI_API_KEY'), 'model': 'gemini-pro'})
    
    vn = MyVanna()
    
    try:
        vn.connect_to_bigquery(project_id='hitech-dados')
        
        # Training data
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

# Cache the SQL generation
@st.cache_data(show_spinner="Gerando consulta SQL...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question)

# Cache the SQL execution
@st.cache_data(show_spinner="Executando consulta...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
    return vn.run_sql(sql=sql)

# Cache the answer generation
@st.cache_data(show_spinner="Gerando explicação...")
def generate_answer_cached(question: str, sql: str, df):
    vn = setup_vanna()
    return vn.generate_answer(question=question, sql=sql, df=df)

# Create the Streamlit interface
query = st.text_input("Digite sua pergunta sobre os dados:")

if query:
    sql = generate_sql_cached(query)
    
    st.subheader("Consulta SQL gerada:")
    st.code(sql, language="sql")
    
    if st.button("Executar consulta"):
        try:
            df = run_sql_cached(sql)
            st.subheader("Resultados:")
            st.dataframe(df)
            
            # Generate explanation
            explanation = generate_answer_cached(query, sql, df)
            st.subheader("Explicação:")
            st.write(explanation)
        except Exception as e:
            st.error(f"Erro ao executar a consulta: {e}")

# Display training data if requested
if st.sidebar.checkbox("Mostrar dados de treinamento"):
    vn = setup_vanna()
    training_data = vn.get_training_data()
    st.sidebar.json(training_data)