from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
import os
from dotenv import load_dotenv
import streamlit as st
import json
import tempfile

# Load environment variables from .env file
load_dotenv()

GCP_PROJECT_ID = 'hitech-dados'
BQ_DATASET = 'seat'
GCP_REGION = "us-central1"

class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        # Set up Google credentials from Streamlit secrets if available
        if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
            # Create a temporary file to store the credentials
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                json.dump(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"], temp_file)
                temp_file_path = temp_file.name
            
            # Set the environment variable to point to the temporary file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
        
        # Initialize ChromaDB
        ChromaDB_VectorStore.__init__(self, config=config)
        
        # Use Streamlit secrets for API key
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv('GEMINI_API_KEY'))
        GoogleGeminiChat.__init__(self, config={'api_key': api_key, 'model': 'gemini-pro'})

# Initialize Vanna
def initialize_vanna():
    vn = MyVanna()
    vn.connect_to_bigquery(project_id=GCP_PROJECT_ID)
    
    # DDL statements are powerful because they specify table names, colume names, types, and potentially relationships
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
        unarchive_history jsonb DEFAULT '[]'::jsonb NULL,
        CONSTRAINT allocations_pkey PRIMARY KEY (id),
        CONSTRAINT fk_rails_21506be62a FOREIGN KEY (xp_manager_id) REFERENCES public.users(id),
        CONSTRAINT fk_rails_6e1b8aff03 FOREIGN KEY (resource_id) REFERENCES public.resources(id),
        CONSTRAINT fk_rails_7166cc3d9c FOREIGN KEY (client_id) REFERENCES public.clients(id),
        CONSTRAINT fk_rails_9cf329e68c FOREIGN KEY (opportunity_id) REFERENCES public.opportunities(id),
        CONSTRAINT fk_rails_ac478fcffe FOREIGN KEY (project_id) REFERENCES public.projects(id),
        CONSTRAINT fk_rails_e570e8d22c FOREIGN KEY (replacement_id) REFERENCES public.resources(id)
    );
    CREATE INDEX index_allocations_on_client_id ON public.allocations USING btree (client_id);
    CREATE INDEX index_allocations_on_opportunity_id ON public.allocations USING btree (opportunity_id);
    CREATE INDEX index_allocations_on_project_id ON public.allocations USING btree (project_id);
    CREATE INDEX index_allocations_on_replacement_id ON public.allocations USING btree (replacement_id);
    CREATE INDEX index_allocations_on_resource_id ON public.allocations USING btree (resource_id);
    CREATE UNIQUE INDEX index_allocations_on_uuid ON public.allocations USING btree (uuid);
    CREATE INDEX index_allocations_on_xp_manager_id ON public.allocations USING btree (xp_manager_id);
    """)
    
    # Sometimes you may want to add documentation about your business terminology or definitions.
    vn.train(documentation="A tabela 'allocations' contém informações sobre os impusers alocados. Quando a alocação iniciou e terminou, em qual projetos estão alocados. Valor hora que o impulser recebe e o valor que a empresa recebe por esse impulser. Qual o cliente o impulser está alocado. Por qual oportunidade de emprego ele foi aprovado e alocado. e qual funcionário dez o acompanhamento.")
    
    # You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
    vn.train(question="Quantos impulsers temos alocados hoje?",
             sql=f"SELECT count(*) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` WHERE end_at >= CURRENT_DATE()")
    
    vn.train(question="Quantos impulsers iniciaram suas alocações em abril de 2025?",
             sql=f"SELECT count(*) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` WHERE start_at >= '2025-04-01'")
    
    vn.train(question="Qual é a porcentagem de fee por impulser que foram alocados a partir de em abril de 2025?",
             sql=f"SELECT round((1 - a.resource_rate / (a.billing_rate * (1-0.1656)))*100,2) as percent_fee FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` a WHERE start_at >= '2025-04-01'")
    
    vn.train(question="Quantos projetos receberam impulsers alocados no mes de abril de 2025?",
             sql=f"SELECT DISTINCT COUNT(project_id) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` WHERE start_at >= '2025-04-01'")
    
    return vn

# Create a global instance of Vanna
vn = initialize_vanna()

# For Flask app if needed
if __name__ == "__main__":
    from vanna.flask import VannaFlaskApp
    app = VannaFlaskApp(vn)
    app.run()