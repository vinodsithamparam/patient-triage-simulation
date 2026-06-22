
# 🏥 Clinical Triage Control Center

The **Clinical Triage Control Center** is an isolated, local multi-container intelligent application built to streamline emergency medical intake and risk management. It combines deterministic validation, machine learning predictive inference and structured LLM clinical assessment.

---

## 🏗️ System Architecture

The application runs across three primary services containerized via Docker:

* **`frontend` (Streamlit Portal)**: An interactive web dashboard allowing medical personnel to register emergency vitality measurements, send payloads to the graph engine, and monitor active patient queues dynamically.
* **`graph` (LangGraph Engine)**: The core intelligent workflow router orchestrates data validation, statistical ML calculations, and structured LLM syntheses.
* **`postgres` (Relational Storage)**: A secure PostgreSQL 15 database tracks clinical session payloads, triage outcomes, and audit registry metrics natively.
---

## 🔁 LangGraph Execution Pipeline

The core analytical pipeline steps sequentially through the following logical graph node array to calculate outcomes:

```
  [START] 
     │
     ▼
┌──────────┐         ❌ (Invalid Vitals)         ┌───────────────┐
│ Validate ├────────────────────────────────────>│ Recount Vitals│
└────┬─────┘                                     └───────┬───────┘
     │                                                   │
     │  ✔ (Data Safe)                                    │ (Corrected)
     ▼                                                   │
┌──────────┐                                             │
│ Predict  │<────────────────────────────────────────────┘
└────┬─────┘
     │
     ├─► [Risk Level == "Red"]    ──► Critical Subgraph ──► [Emergency ICU Team]   ─┐
     │                                                                              │
     └─► [Risk Level != "Red"]    ──► Standard Subgraph ──► [Standard Ward Team]  ─┼─► [DB Update] ──► [END]

```

1. **Validation Node (`validate`)**: Cross-checks incoming vitals data (such as SpO2 and Heart Rate bounds) to catch measurement anomalies or logical errors.
2. **Human-in-the-Loop Recount Node (`recount_vitals`)**: Intercepts in the event of unrealistic vitals data, pausing the state machine to re-measure vitals before releasing the flow to inference.
3. **Predictive Inference Node (`predict`)**: Employs an XGBoost model and SHAP tree explainer to compute localized patient risk probabilities and isolate key physiological driver metrics.
4. **Conditional Subgraph Routing**:
    * **Critical Path**: Triggers automated stat alerts and runs structured clinical note analysis via an integrated LLM node, automatically dispatching the patient to the **Emergency ICU Team**.
    * **Standard Path**: Performs standard feedback processing, and assigns them to the **Standard Ward Team**.
5. **Database Update Node (`db_update`)**: Stores the final patient state to a persistent database layer.

---

## 🚀 Local Deployment Instructions

### 1. Prerequisites

Ensure you have the following software infrastructure platforms configured on your local host machine:

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine
* Docker Compose v2 plugin (native Go binary format)

### 2. Project Directory Organization

Confirm your repository context directory structure perfectly matches the following baseline template layout:

```text
.
├── docker-compose.yml
├── .env
├── .gitignore
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── graph/
│   ├── Dockerfile
│   ├── langgraph.json
│   ├── preprocessor.joblib
│   ├── requirements.txt
│   ├── triage_graph.py
│   └── xgb_model.joblib
└── postgres/
    └── init.sql

```

### 3. Environment Setup Configuration

Create a file named `.env` at your **root project directory** level (right alongside `docker-compose.yml`). Populate it with your target access parameters and platform credentials:

```env
# Large Language Model Engine Credentials
LANGSMITH_API_KEY=gsk_your_actual_langsmith_api_key_string
GROQ_LLM=qwen/qwen3-32b
GROQ_API_KEY=gsk_your_actual_groq_api_key_string
GRAPH_NAME=triage
DB_HOST=postgres
DB_USER=postgres
DB_PASS=password
DB_NAME=triage_db

```

### 4. Running the Application Stack

Execute the absolute system build sequence from your root path using your terminal shell command runner:

```bash
docker compose up --build

```

Docker will build your Python 3.12 containers, map port mappings outward to the host system layer, establish internal dependencies, and run your database schema files (`init.sql`) to initialize the required tables.

---

## 📊 Verifying and Accessing Your Application

Once the logs print that all components have fully initialized, navigate to the following uniform web portal landing pages:

* **Interactive Streamlit Dashboard**: Open your browser and go to `http://localhost:8501`. Fill out patient reference forms and inspect metrics live.
* **LangGraph Development Endpoint**: Access your underlying server pipeline directly at `http://localhost:8123`.
* **LangSmith Studio Interface**: If you wish to visually debug your graph steps, connect your local deployment engine directly using the [LangSmith Studio UI Portal](https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:8123).

### Inspecting Local Database Storage

If you wish to log directly into the running Postgres engine to run SQL test scripts against your saved registry items, execute this command inside a new host shell terminal window:

```bash
docker exec -it triage_postgres psql -U postgres -d triage_db
```

To see what records have been successfully added to the system database, run:

```sql
SELECT patient_id, risk_level, risk_score, triage_urgency FROM triage_sessions;
```

*(Type `\q` to cleanly exit the psql environment when you are done).*
