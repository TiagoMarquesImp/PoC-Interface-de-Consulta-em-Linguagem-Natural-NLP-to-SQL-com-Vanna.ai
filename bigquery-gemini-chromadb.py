# Add this at the very top of your file, before any other imports
import os
import sys
import json

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
from dotenv import load_dotenv
from google.oauth2 import service_account

# Load environment variables from .env file
load_dotenv()

GCP_PROJECT_ID = 'hitech-dados'
BQ_DATASET = 'seat'
GCP_REGION = "us-central1"

# Create a credentials.json file from environment variable if it exists
credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
credentials_path = None

if credentials_json:
    # Create a temporary credentials file
    credentials_path = "google_credentials.json"
    with open(credentials_path, 'w') as f:
        f.write(credentials_json)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        GoogleGeminiChat.__init__(self, config={'api_key': os.getenv('GEMINI_API_KEY'), 'model': 'gemini-pro'})

vn = MyVanna()

# Modified connection to BigQuery with explicit credentials if available
if credentials_path:
    vn.connect_to_bigquery(project_id=GCP_PROJECT_ID, cred_file_path=credentials_path)
else:
    # Try to connect using default credentials (for local development)
    try:
        vn.connect_to_bigquery(project_id=GCP_PROJECT_ID)
    except Exception as e:
        print(f"Failed to connect to BigQuery: {e}")
        print("Please set GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

# The following are methods for adding training data. Make sure you modify the examples to match your database.

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
);
""")

# Sometimes you may want to add documentation about your business terminology or definitions.
vn.train(documentation="A tabela 'allocations' contém informações sobre os impusers alocados. Quando a alocação iniciou e terminou, em qual projetos estão alocados. Valor hora que o impulser recebe e o valor que a empresa recebe por esse impulser. Qual o cliente o impulser está alocado. Por qual oportunidade de emprego ele foi aprovado e alocado. e qual funcionário dez o acompanhamento.")

# You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
vn.train(question="Quantos impulsers temos alocados hoje?",
         sql=f"select count(*) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations` where end_at >= current_date")

vn.train(question="Quantos impulsers iniciaram suas alocações em abril de 2025?",
         sql=f"select count(*) FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations`  where start_at >= '2025-04-01")

vn.train(question="Qual é a porcentagem de fee por impulser que foram alocados a partir de em abril de 2025?",
         sql=f"select round((1 - a.resource_rate  / (a.billing_rate  * (1-0.1656)))*100,2) as percent_fee FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.allocations`  where start_at >= '2025-04-01")

vn.train(question="Quantos projetos receberam impulsers alocados no mes de abril de 2025?",
         sql=f"""
             select distinct count(project_id)
             FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.pedidos` AS p
             where start_at >= '2025-04-01'
         """)

# At any time you can inspect what training data the package is able to reference
training_data = vn.get_training_data()


from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)
app.run()