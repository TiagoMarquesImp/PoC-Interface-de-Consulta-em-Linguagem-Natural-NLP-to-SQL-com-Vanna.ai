# Add this at the very top of your file, before any other imports
import os
import sys
import streamlit as st # <-- Importa o Streamlit
import json
import time
import plotly.express as px
import plotly.graph_objects as go # Renomeado para go para evitar conflito com fig no exec

# --- MOVER st.set_page_config() PARA CÁ ---
# Deve ser o PRIMEIRO comando Streamlit após os imports
st.set_page_config(page_title="Consulta em Linguagem Natural - Impulso", layout="wide")
# --------------------------------------------

# Tentativa de importar e configurar pysqlite3 para ambientes como Streamlit Cloud
# Agora esta parte vem DEPOIS de set_page_config
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    # Nota: Chamar st.sidebar aqui no início pode ter um efeito visual mínimo
    # Se preferir, pode mover essas mensagens para dentro da função setup_vanna
    st.sidebar.info("SQLite driver substituído por pysqlite3.")
except ImportError:
    st.sidebar.warning("pysqlite3 não encontrado, usando sqlite3 padrão.")
    # pass # Continua com o sqlite3 padrão se pysqlite3 não estiver disponível

# --- Outros imports necessários ---
from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import tempfile # Importado para usar no path do ChromaDB

# --- Constantes do Projeto ---
GCP_PROJECT_ID = 'hitech-dados'
BQ_DATASET = 'seat'
# GCP_REGION = "us-central1" # Não parece ser usado diretamente no código Vanna aqui

# Set up the Streamlit page
st.set_page_config(page_title="Consulta em Linguagem Natural - Impulso", layout="wide")

# Initialize Vanna with caching using Streamlit Secrets
@st.cache_resource(ttl=3600)
def setup_vanna():
    st.sidebar.info("Tentando configurar Vanna com Streamlit Secrets...") # Log inicial

    # --- Obter Credenciais do Google Cloud (BigQuery) ---
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" not in st.secrets:
        st.sidebar.error("❌ Secret 'GOOGLE_APPLICATION_CREDENTIALS_JSON' não encontrado!")
        st.error("Erro de Configuração: O secret com as credenciais JSON do Google Cloud não foi encontrado.")
        st.stop() # Interrompe a execução se a credencial estiver faltando

    try:
        credentials_json_str = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
        # Verifica se a string não está vazia
        if not credentials_json_str:
             st.sidebar.error("❌ Secret 'GOOGLE_APPLICATION_CREDENTIALS_JSON' está vazio!")
             st.error("Erro de Configuração: O secret 'GOOGLE_APPLICATION_CREDENTIALS_JSON' está vazio.")
             st.stop()
        credentials_info = json.loads(credentials_json_str)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        st.sidebar.success("✅ Credenciais BigQuery carregadas dos secrets.")
    except json.JSONDecodeError as e:
        st.sidebar.error(f"❌ Erro ao decodificar JSON das credenciais do Google Cloud: {e}")
        st.error("Erro de Configuração: O conteúdo do secret 'GOOGLE_APPLICATION_CREDENTIALS_JSON' não é um JSON válido.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao processar credenciais BigQuery: {e}")
        st.error(f"Erro inesperado ao processar credenciais BigQuery: {e}")
        st.stop()

    # --- Obter Chave da API Gemini ---
    if "GEMINI_API_KEY" not in st.secrets:
        st.sidebar.error("❌ Secret 'GEMINI_API_KEY' não encontrado!")
        st.error("Erro de Configuração: O secret com a chave da API Gemini não foi encontrado.")
        st.stop() # Interrompe se a chave estiver faltando

    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    if not gemini_api_key:
         st.sidebar.error("❌ Secret 'GEMINI_API_KEY' está vazio!")
         st.error("Erro de Configuração: O secret 'GEMINI_API_KEY' está vazio.")
         st.stop()
    st.sidebar.success("✅ Chave API Gemini carregada dos secrets.")

    # --- Criar Cliente BigQuery ---
    try:
        client = bigquery.Client(credentials=credentials, project=GCP_PROJECT_ID)
        # Testa a conexão fazendo uma chamada simples (opcional, mas recomendado)
        client.list_datasets(max_results=1)
        st.sidebar.info(f"✅ Cliente BigQuery criado e conexão testada para o projeto '{GCP_PROJECT_ID}'.")
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao criar/testar cliente BigQuery: {e}")
        st.error(f"Erro ao conectar ao BigQuery com as credenciais fornecidas: {e}")
        st.stop()

    # --- Inicializar Vanna ---
    # Passa a chave da API Gemini diretamente na inicialização
    class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
        def __init__(self, config=None):
            # Configuração do ChromaDB (pode precisar ajustar o caminho se necessário no Streamlit)
            # Usar um path temporário pode ser uma opção em ambientes restritos
            chroma_path = os.path.join(tempfile.gettempdir(), "vanna_chroma_db")
            st.sidebar.info(f"Usando path ChromaDB: {chroma_path}")
            # Certifique-se que o diretório exista
            os.makedirs(chroma_path, exist_ok=True)
            chroma_config = {'path': chroma_path}

            ChromaDB_VectorStore.__init__(self, config=chroma_config)
            GoogleGeminiChat.__init__(self, config={'api_key': gemini_api_key, 'model': 'gemini-pro'})

    vn = MyVanna()
    st.sidebar.info("Instância Vanna (MyVanna) inicializada.")

    # --- Conectar Vanna ao BigQuery (Usando o cliente já criado) ---
    # Atribui o cliente BigQuery diretamente à instância Vanna
    vn.bigquery_client = client
    vn.bigquery_project_id = GCP_PROJECT_ID # Define o project ID também se necessário
    st.sidebar.success("✅ Instância Vanna conectada ao BigQuery usando cliente dos secrets.")

    # --- Treinamento Vanna (permanece igual) ---
    try:
        st.sidebar.info("Iniciando treinamento Vanna...")
        # Verifica se já existe algum dado de treinamento para evitar duplicar
        existing_training_data = vn.get_training_data()
        if not existing_training_data: # Treina apenas se não houver dados existentes
            st.sidebar.warning("Nenhum dado de treinamento encontrado. Iniciando treinamento...")

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

            st.sidebar.success("✅ Treinamento Vanna concluído.")
        else:
            st.sidebar.info("✅ Dados de treinamento já existem. Pulando treinamento.")

    except Exception as e:
        st.sidebar.error(f"❌ Erro durante o treinamento Vanna: {e}")
        st.error(f"Erro durante o treinamento Vanna: {e}")
        # Considere parar aqui se o treinamento for crítico
        # st.stop()

    return vn

# --- Funções Cacheadas (Baseadas no antigo vanna_calls.py e app.py) ---

@st.cache_data(ttl=3600)
def generate_questions_cached():
    vn = setup_vanna()
    try:
        return vn.generate_questions()
    except Exception as e:
        st.error(f"Erro ao gerar perguntas sugeridas: {e}")
        return []

@st.cache_data(show_spinner="Gerando consulta SQL...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    try:
        return vn.generate_sql(question=question)
    except Exception as e:
        st.error(f"Erro ao gerar SQL: {e}")
        return None # Retorna None em caso de erro

@st.cache_data(show_spinner="Executando consulta...")
def run_sql_cached(sql: str):
    if not sql: # Verifica se o SQL não é None ou vazio
        st.warning("Nenhuma consulta SQL para executar.")
        return None
    vn = setup_vanna()
    try:
        df = vn.run_sql(sql=sql)
        # st.success("Consulta SQL executada com sucesso.") # Log opcional
        return df
    except Exception as e:
        st.error(f"Erro ao executar SQL no BigQuery: {e}")
        # Tenta dar uma dica sobre o erro de conexão, se aplicável
        if "NoneType" in str(e) or "connect" in str(e).lower():
            st.warning("Dica: Verifique a configuração das credenciais BigQuery nos secrets do Streamlit.")
        return None # Retorna None em caso de erro

@st.cache_data(show_spinner="Gerando código para visualização...")
def generate_plotly_code_cached(question: str, sql: str, df):
    if df is None or df.empty:
        return None # Não gera código se não houver dados
    vn = setup_vanna()
    try:
        return vn.generate_plotly_code(question=question, sql=sql, df=df)
    except Exception as e:
        st.error(f"Erro ao gerar código Plotly: {e}")
        return None

@st.cache_data(show_spinner="Gerando visualização...")
def generate_plot_cached(code: str, df):
    if not code or df is None or df.empty:
        return None # Não gera gráfico sem código ou dados
    try:
        # Usa go (plotly.graph_objects) importado no início
        local_vars = {"px": px, "go": go, "df": df}
        exec(code, {"__builtins__": {}}, local_vars) # Ambiente de execução mais seguro
        fig = local_vars.get("fig")
        if fig is None:
             st.warning("Código Plotly executado, mas não gerou um objeto 'fig'.")
        # else:
        #     st.success("Gráfico gerado com sucesso.") # Log opcional
        return fig
    except Exception as e:
        st.error(f"Erro ao executar código Plotly e gerar gráfico: {e}\nCódigo:\n```python\n{code}\n```")
        return None

@st.cache_data(show_spinner="Gerando perguntas relacionadas...")
def generate_followup_cached(question: str, sql: str, df):
    if df is None: # Pode gerar followup mesmo sem df, mas talvez menos relevante
        df = pd.DataFrame() # Passa um DF vazio para evitar erros
    vn = setup_vanna()
    try:
        return vn.generate_followup_questions(question=question, sql=sql, df=df)
    except Exception as e:
        st.error(f"Erro ao gerar perguntas relacionadas: {e}")
        return []

@st.cache_data(show_spinner="Verificando se devo gerar gráfico...")
def should_generate_chart_cached(question: str, sql: str, df):
    if df is None or df.empty:
        return False # Não gera gráfico sem dados
    vn = setup_vanna()
    try:
        return vn.should_generate_chart(question=question, sql=sql, df=df)
    except Exception as e:
        st.error(f"Erro ao verificar se gráfico deve ser gerado: {e}")
        return False

@st.cache_data(show_spinner="Verificando SQL...")
def is_sql_valid_cached(sql: str):
    # Validação simples - pode ser aprimorada se necessário
    # O próprio Vanna pode ter uma função de validação mais robusta
    if not isinstance(sql, str):
         return False
    # Verifica se contém SELECT e não contém comandos perigosos básicos
    sql_lower = sql.lower()
    if "select" not in sql_lower:
        return False
    if any(keyword in sql_lower for keyword in ["delete", "update", "drop", "insert", "alter"]):
        st.warning("SQL parece conter comandos de modificação/exclusão.")
        return False # Considera inválido por segurança
    # Tentar usar a validação interna do Vanna se disponível e confiável
    # try:
    #     vn = setup_vanna()
    #     return vn.is_sql_valid(sql=sql)
    # except Exception as e:
    #     st.warning(f"Erro na validação de SQL interna do Vanna: {e}")
    #     return False # Assume inválido em caso de erro na validação interna
    return True # Passa na validação básica

@st.cache_data(show_spinner="Gerando resumo...")
def generate_summary_cached(question: str, df):
    if df is None or df.empty:
        return "Não há dados para resumir." # Mensagem mais clara
    vn = setup_vanna()
    try:
        return vn.generate_summary(question=question, df=df) # Use generate_summary
        # Ou generate_answer se preferir: return vn.generate_answer(question=question, df=df)
    except Exception as e:
        st.error(f"Erro ao gerar resumo: {e}")
        return None

# --- Interface do Usuário Streamlit (Baseada no antigo app.py) ---

# Inicializa o estado da sessão se não existir
if "my_question" not in st.session_state:
    st.session_state["my_question"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None

avatar_url = "https://vanna.ai/img/vanna.svg"

st.title("Consulta em Linguagem Natural - Impulso")
st.write("Faça perguntas sobre os dados de alocação de impulsers em linguagem natural")

# Sidebar configuration
st.sidebar.title("Configurações de Saída")
st.sidebar.checkbox("Mostrar SQL", value=True, key="show_sql")
st.sidebar.checkbox("Mostrar Tabela", value=True, key="show_table")
st.sidebar.checkbox("Mostrar Código Plotly", value=False, key="show_plotly_code")
st.sidebar.checkbox("Mostrar Gráfico", value=True, key="show_chart")
st.sidebar.checkbox("Mostrar Resumo", value=True, key="show_summary")
st.sidebar.checkbox("Mostrar Perguntas Relacionadas", value=True, key="show_followup")

def reset_question():
    st.session_state["my_question"] = None
    st.session_state["df"] = None # Limpa o dataframe também

st.sidebar.button("Nova Pergunta", on_click=reset_question, use_container_width=True)

# Display training data if requested
if st.sidebar.checkbox("Mostrar dados de treinamento (se houver)"):
    try:
        vn = setup_vanna() # Garante que Vanna está configurado
        training_data = vn.get_training_data()
        if training_data:
            st.sidebar.json(training_data)
        else:
            st.sidebar.info("Nenhum dado de treinamento encontrado na instância Vanna.")
    except Exception as e:
        st.sidebar.error(f"Erro ao obter dados de treinamento: {e}")


def set_question(question):
    st.session_state["my_question"] = question

# Suggested questions
# Só mostra o botão se não houver pergunta ativa
if not st.session_state.get("my_question"):
    assistant_message_suggested = st.chat_message("assistant", avatar=avatar_url)
    if assistant_message_suggested.button("Mostrar perguntas sugeridas"):
        # st.session_state["my_question"] = None # Já é None aqui
        questions = generate_questions_cached()
        if questions:
            for question in questions:
                time.sleep(0.05)
                st.button(question, on_click=set_question, args=(question,))
        else:
            st.info("Não foi possível gerar perguntas sugeridas no momento.")


# Get question from session state or input
# Usa a chave 'my_question' consistentemente
my_question = st.session_state.get("my_question")

if my_question is None:
    my_question = st.chat_input("Faça uma pergunta sobre seus dados...")
    if my_question: # Se o usuário digitou algo novo
         st.session_state["my_question"] = my_question
         st.rerun() # Roda novamente para processar a nova pergunta

# Process the question if it exists
if my_question:
    # Mostra a pergunta do usuário
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    # Gera SQL
    sql = generate_sql_cached(question=my_question)

    if sql:
        # Valida SQL (opcional, mas recomendado)
        if not is_sql_valid_cached(sql=sql):
            assistant_message = st.chat_message("assistant", avatar=avatar_url)
            assistant_message.error(f"O SQL gerado parece inválido ou inseguro:\n```sql\n{sql}\n```")
            st.stop() # Para a execução se o SQL for inválido

        # Mostra SQL se configurado
        if st.session_state.get("show_sql", True):
            assistant_message_sql = st.chat_message("assistant", avatar=avatar_url)
            assistant_message_sql.code(sql, language="sql", line_numbers=True)

        # Executa SQL
        # Tenta usar o dataframe do cache se a pergunta for a mesma
        # No entanto, run_sql_cached já faz cache, então podemos chamá-lo diretamente
        df = run_sql_cached(sql=sql)

        # Armazena o dataframe no estado da sessão se for bem-sucedido
        if df is not None:
            st.session_state["df"] = df
        else:
            # Se run_sql falhou, a mensagem de erro já foi exibida na função _cached
            # Podemos parar aqui ou deixar continuar para tentar mostrar followups, etc.
             st.warning("A execução do SQL falhou. Não é possível prosseguir com tabela, gráfico ou resumo.")
             # Limpa df do estado para garantir que não use dados antigos
             st.session_state["df"] = None
             # st.stop() # Descomente se quiser parar totalmente aqui


        # Processa o dataframe se ele existir no estado da sessão
        if st.session_state.get("df") is not None:
            df_results = st.session_state.get("df") # Pega o df do estado

            # Mostra Tabela se configurado
            if st.session_state.get("show_table", True):
                assistant_message_table = st.chat_message("assistant", avatar=avatar_url)
                if not df_results.empty:
                    assistant_message_table.write("Resultados da Consulta:")
                    # Mostra os 10 primeiros ou todos se forem poucos
                    if len(df_results) > 10:
                        assistant_message_table.dataframe(df_results.head(10))
                        assistant_message_table.caption(f"Mostrando 10 de {len(df_results)} linhas.")
                    else:
                        assistant_message_table.dataframe(df_results)
                else:
                    assistant_message_table.info("A consulta SQL não retornou resultados.")

            # Gera e mostra Gráfico se configurado e aplicável
            if not df_results.empty and st.session_state.get("show_chart", True):
                if should_generate_chart_cached(question=my_question, sql=sql, df=df_results):
                    code = generate_plotly_code_cached(question=my_question, sql=sql, df=df_results)

                    # Mostra Código Plotly se configurado
                    if code and st.session_state.get("show_plotly_code", False):
                        assistant_message_plotly_code = st.chat_message("assistant", avatar=avatar_url)
                        assistant_message_plotly_code.code(code, language="python", line_numbers=True)

                    # Gera e mostra o gráfico
                    if code:
                        assistant_message_chart = st.chat_message("assistant", avatar=avatar_url)
                        fig_plot = generate_plot_cached(code=code, df=df_results)
                        if fig_plot is not None:
                            assistant_message_chart.plotly_chart(fig_plot)
                        else:
                            # Mensagem de erro já exibida em generate_plot_cached
                            pass # assistant_message_chart.error("Não foi possível gerar um gráfico")
                    # else:
                        # st.info("Não foi gerado código para o gráfico.") # Log opcional
                # else:
                    # st.info("Gráfico não recomendado para esta consulta/resultado.") # Log opcional


            # Gera e mostra Resumo se configurado
            if not df_results.empty and st.session_state.get("show_summary", True):
                assistant_message_summary = st.chat_message("assistant", avatar=avatar_url)
                summary = generate_summary_cached(question=my_question, df=df_results)
                if summary:
                    assistant_message_summary.write(summary)
                # else:
                    # Mensagem de erro/aviso já exibida em generate_summary_cached
                    # pass

            # Gera e mostra Perguntas Relacionadas se configurado
            if st.session_state.get("show_followup", True):
                assistant_message_followup = st.chat_message("assistant", avatar=avatar_url)
                # Passa df_results ou um df vazio se preferir não basear em resultados
                followup_questions = generate_followup_cached(question=my_question, sql=sql, df=df_results if df_results is not None else pd.DataFrame())

                if followup_questions:
                    assistant_message_followup.write("Aqui estão algumas perguntas relacionadas:")
                    # Mostra as 5 primeiras
                    for question in followup_questions[:5]:
                        # Usa uma chave única para cada botão de followup
                        # f"followup_{question[:20]}" é uma forma simples de tentar criar chaves únicas
                        st.button(question, key=f"followup_{question[:20]}", on_click=set_question, args=(question,))
                # else:
                    # assistant_message_followup.info("Não foram geradas perguntas relacionadas.") # Log opcional

            # Limpa o dataframe do estado da sessão após processar followups
            # para garantir que uma nova pergunta não use dados antigos acidentalmente
            # st.session_state["df"] = None # Movido para o botão Reset

    else:
        # Se a geração de SQL falhou
        assistant_message_error = st.chat_message("assistant", avatar=avatar_url)
        # A mensagem de erro já deve ter sido mostrada em generate_sql_cached
        assistant_message_error.warning("Não foi possível gerar uma consulta SQL para essa pergunta.")
