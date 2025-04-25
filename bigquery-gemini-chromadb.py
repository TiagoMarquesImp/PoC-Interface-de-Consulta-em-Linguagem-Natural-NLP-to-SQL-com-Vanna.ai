from vanna.chromadb import ChromaDB_VectorStore
from vanna.google import GoogleGeminiChat
import os
from dotenv import load_dotenv
import json
import tempfile

# Try to load environment variables from .env file
load_dotenv()

# Try to load from Streamlit secrets if available
try:
    import streamlit as st
    # Check if we're running in Streamlit
    if 'GEMINI_API_KEY' in st.secrets:
        gemini_api_key = st.secrets['GEMINI_API_KEY']
        
        # Create a temporary credentials file from the JSON in secrets
        if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in st.secrets:
            # Create a temporary file to store the credentials
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                temp_file.write(st.secrets['GOOGLE_APPLICATION_CREDENTIALS_JSON'].encode())
                temp_credentials_path = temp_file.name
            
            # Set the environment variable to point to our temp file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_path
    else:
        # Fall back to .env file
        gemini_api_key = os.getenv('GEMINI_API_KEY')
except ImportError:
    # Not running in Streamlit, use environment variables
    gemini_api_key = os.getenv('GEMINI_API_KEY')

GCP_PROJECT_ID = 'hitech-dados'
BQ_DATASET = 'seat'
GCP_REGION = "us-central1"



class MyVanna(ChromaDB_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        GoogleGeminiChat.__init__(self, config={'api_key': gemini_api_key, 'model': 'gemini-pro'})

vn = MyVanna()

# Connect to BigQuery using the credentials from environment or Streamlit secrets
vn.connect_to_bigquery(project_id='hitech-dados')

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

# Create a custom Flask app instead of using VannaFlaskApp
from flask import Flask, render_template, request, jsonify
import uuid
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

# Store chat history
chat_history = []

@app.route('/')
def home():
    return render_template('index.html', chat_history=chat_history)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Get SQL from Vanna
    sql = vn.generate_sql(question)
    
    # Execute the SQL
    try:
        df = vn.run_sql(sql)
        result = df.to_dict(orient='records')
        explanation = vn.explain_sql(sql)
        
        # Add to chat history
        chat_entry = {
            'question': question,
            'sql': sql,
            'result': result,
            'explanation': explanation
        }
        chat_history.append(chat_entry)
        
        return jsonify(chat_entry)
    except Exception as e:
        return jsonify({'error': str(e), 'sql': sql}), 500

# Create templates directory and HTML file if they don't exist
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

# Create the HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>SQL Chat Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f5f5f5; }
        .chat-container { 
            height: 70vh; 
            overflow-y: auto; 
            margin-bottom: 20px; 
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .user-message { 
            background-color: #e3f2fd; 
            padding: 10px; 
            border-radius: 10px; 
            margin: 10px 0; 
            max-width: 80%;
            margin-left: auto;
        }
        .bot-message { 
            background-color: #f1f1f1; 
            padding: 10px; 
            border-radius: 10px; 
            margin: 10px 0; 
            max-width: 80%;
        }
        .sql-code { 
            background-color: #272822; 
            color: #f8f8f2; 
            padding: 10px; 
            border-radius: 5px; 
            overflow-x: auto; 
            font-family: monospace;
        }
        .result-table { 
            margin-top: 10px; 
            overflow-x: auto; 
            background-color: white;
            border-radius: 5px;
        }
        .loading { 
            display: none; 
            align-items: center;
            justify-content: center;
            margin-top: 10px;
        }
        .header {
            background-color: #0d6efd;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="mb-0">SQL Chat Assistant</h1>
            <p class="mb-0">Ask questions about your data in natural language</p>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <!-- Chat messages will appear here -->
            <div class="bot-message">
                <strong>Assistant:</strong> Hello! I'm your SQL assistant. Ask me questions about your data and I'll generate SQL queries to answer them.
            </div>
        </div>
        
        <div class="input-group mb-3">
            <input type="text" id="questionInput" class="form-control" placeholder="Ask a question about your data...">
            <button class="btn btn-primary" id="askButton">Ask</button>
        </div>
        
        <div class="loading" id="loadingIndicator">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <span class="ms-2">Generating SQL and fetching results...</span>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chatContainer');
            const questionInput = document.getElementById('questionInput');
            const askButton = document.getElementById('askButton');
            const loadingIndicator = document.getElementById('loadingIndicator');
            
            // Function to add a message to the chat
            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = isUser ? 'user-message' : 'bot-message';
                messageDiv.innerHTML = content;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            // Function to handle asking a question
            async function askQuestion() {
                const question = questionInput.value.trim();
                if (!question) return;
                
                // Add user question to chat
                addMessage(`<strong>You:</strong> ${question}`, true);
                
                // Clear input and show loading
                questionInput.value = '';
                loadingIndicator.style.display = 'flex';
                askButton.disabled = true;
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question }),
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Create response HTML
                        let responseHTML = `<strong>Assistant:</strong><br>`;
                        
                        // Add SQL
                        responseHTML += `<p>Generated SQL:</p><pre class="sql-code">${data.sql}</pre>`;
                        
                        // Add explanation if available
                        if (data.explanation) {
                            responseHTML += `<p>Explanation: ${data.explanation}</p>`;
                        }
                        
                        // Add results as table if available
                        if (data.result && data.result.length > 0) {
                            responseHTML += `<div class="result-table"><table class="table table-striped table-sm">`;
                            
                            // Table headers
                            responseHTML += '<thead><tr>';
                            for (const key of Object.keys(data.result[0])) {
                                responseHTML += `<th>${key}</th>`;
                            }
                            responseHTML += '</tr></thead>';
                            
                            // Table body
                            responseHTML += '<tbody>';
                            for (const row of data.result) {
                                responseHTML += '<tr>';
                                for (const key of Object.keys(row)) {
                                    responseHTML += `<td>${row[key]}</td>`;
                                }
                                responseHTML += '</tr>';
                            }
                            responseHTML += '</tbody></table></div>';
                        } else {
                            responseHTML += '<p>No results returned.</p>';
                        }
                        
                        addMessage(responseHTML);
                    } else {
                        // Handle error
                        addMessage(`<strong>Error:</strong> ${data.error}<br>Generated SQL: <pre class="sql-code">${data.sql || 'None'}</pre>`);
                    }
                } catch (error) {
                    addMessage(`<strong>Error:</strong> ${error.message}`);
                } finally {
                    loadingIndicator.style.display = 'none';
                    askButton.disabled = false;
                }
            }
            
            // Event listeners
            askButton.addEventListener('click', askQuestion);
            questionInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        });
    </script>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    ''')

if __name__ == '__main__':
    app.run(debug=True, port=5000)


