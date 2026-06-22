import os
import re
import joblib
import pandas as pd
import numpy as np
import shap
import sqlalchemy
from typing import TypedDict, List, Dict, Literal, NotRequired
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START 

def connect_local_postgres() -> sqlalchemy.engine.base.Engine:
    """Initializes a traditional connection pool targeting standard local network container configurations."""
    db_host = os.environ.get("DB_HOST")
    db_user = os.environ.get("DB_USER")
    db_pass = os.environ.get("DB_PASS")
    db_name = os.environ.get("DB_NAME")
    
    # 1. Retrieve the port configuration fallback
    db_port = os.environ.get("DB_PORT")
    
    # 2. Defensive check: Safeguard against empty values or literal "None" strings
    if not db_port or str(db_port).strip() == "None" or str(db_port).strip() == "":
        db_port = "5432"
    
    # 3. Assemble the explicit connection mapping string safely
    connection_string = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    print(f"🔌 Connecting to local database at string path: postgresql+psycopg2://{db_user}:***@{db_host}:{db_port}/{db_name}")
    return sqlalchemy.create_engine(connection_string)

# Initialize the global engine pool
try:
    if "DB_HOST" in os.environ:
        db_engine = connect_local_postgres()
except Exception as e:
    print(f"⚠️ Database connection framework error payload failed to instantiate: {e}")
    db_engine = None

# State Definition
class PatientTriageSummary(BaseModel):
    triage_urgency: Literal['Low','Medium','High']
    primary_system_affected: Literal['Cardiovascular','Respiratory','Neurological','Gastrointestinal','General']
    symptom_progression: Literal['Improving','Stable','Worsening']
    recommended_action: Literal['Discharge','Admit','ICU', 'Further Testing']
    distress_level: int = Field(ge=1, le=10, description="Sentiment-based distress score")
    potential_diagnosis_category: str = Field(description="Statistical suggestion of diagnosis.")

class PatientVitals(TypedDict):
    Age: int
    Gender: Literal['Male','Female','Other']
    HeartRate_bpm: int
    RespRate_bpm: int
    SystolicBP: int
    DiastolicBP: int
    Temperature_C: float
    SpO2_percent: int
    FastingGlucose_mg_dL: int
    TotalCholesterol_mg_dL: int
    Weight_kg: float
    Height_cm: float
    BMI: float
    Pain_Level_0_to_10: int
    Mental_Status: Literal['Alert','Confused','Lethargic','Unresponsive']
    Symptom_Duration_Days: int

class PatientInput(TypedDict):
    patient_id: str
    vitals: PatientVitals
    doctor_notes: NotRequired[str]
    patient_notes: NotRequired[str]

class OverallState(TypedDict):
    patient_id: str
    patient_notes: str
    doctor_notes: str
    vitals: PatientVitals
    risk_score: float
    risk_level: str
    feature_importance: List[str]
    triage_summary: PatientTriageSummary
    validation_errors: List[str]
    invalid_keys: List[str]
    current_handler: str

# Prompt template
triage_prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert AI clinical triage assistant. Synthesize the provided patient vitals, "
        "doctor notes, and patient notes to complete a structured triage summary. "
        "Be conservative, clinically precise, and select the most appropriate options based on risk data.\n\n"
        "Context Risk Assessment:\n"
        "- Calculated Risk Score: {risk_score}\n"
        "- Risk Level Classification: {risk_level}\n"
        "- Key Risk Drivers: {feature_importance}"
    )),
    ("user", (
        "### Patient Data Input:\n"
        "Vitals: {vitals}\n"
        "Doctor's Clinical Notes: {doctor_notes}\n"
        "Patient/Feedback Notes: {patient_notes}"
    ))
])

# Initialize ChatGroq LLM
llm = ChatGroq(
    model=os.environ["GROQ_LLM"],
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
    include_reasoning=False,
)

structured_llm = llm.with_structured_output(PatientTriageSummary)

# Validate Patient Vitals Node  
def validate_vitals_node(state: PatientInput) -> OverallState:
    vitals = state["vitals"]  
    errors = []
    invalid_keys = []

    if vitals["SpO2_percent"] > 100 or vitals["SpO2_percent"] < 20:
        errors.append(f"Invalid SpO2 reading: {vitals['SpO2_percent']}%")
        invalid_keys.append("SpO2_percent")

    if vitals["HeartRate_bpm"] > 250 or vitals["HeartRate_bpm"] < 0:
        errors.append(f"Heart Rate out of range: {vitals['HeartRate_bpm']} bpm")
        invalid_keys.append("HeartRate_bpm")

    return {"validation_errors": errors, "invalid_keys": invalid_keys}

# Recount Vitals Node 
def recount_vitals_node(state: OverallState) -> OverallState:

    print("\n" + "="*50 + "\n⚠️ RECOUNT TRIGGERED")
    for error in state["validation_errors"]:
        print(f" - {error}")
    print("="*50)

    corrected_vitals = state["vitals"].copy()
    vitals_schema = PatientVitals.__annotations__

    for key in state["invalid_keys"]:
        expected_type = vitals_schema.get(key)
        while True:
            try:
                new_val_str = input(f"Enter corrected value for '{key}' (Expected {expected_type.__name__}): ")
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Literal:
                    new_val = new_val_str
                else:
                    new_val = expected_type(new_val_str)
                corrected_vitals[key] = new_val
                break
            except ValueError:
                print(f"❌ Format error. This field must be a {expected_type.__name__}.")

    return {"vitals": corrected_vitals, "validation_errors": [], "invalid_keys": []}

# Recount Conditional 
def should_recount(state: OverallState) -> str:
    if state["invalid_keys"]:
        return "recount"
    return "proceed"

def predictive_inference_node(state: OverallState) -> Dict:

    print("\n🧠 --- Running ML Predictive Inference ---")

    # Load model and pipeline artifact 
    try:
        print("Loading ML artifacts...")
        preprocessor = joblib.load("preprocessor.joblib")
        xgb_model = joblib.load("xgb_model.joblib")
        print("Model and pre-processor loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠️ Error: Could not find model files. Details: {e}")
        raise
    
    # Transform Patient Input 
    patient_df = pd.DataFrame([state["vitals"]])
    X_processed = preprocessor.transform(patient_df)
    risk_prob = float(xgb_model.predict_proba(X_processed)[0, 1])

    risk_level = "Green"
    if risk_prob >= 0.75:
        risk_level = "Red"
    elif risk_prob >= 0.40:
        risk_level = "Yellow"

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_processed)[0]
    feature_names = preprocessor.get_feature_names_out()

    top_indices = np.argsort(shap_vals)[-2:]
    important_features = [feature_names[i] for i in reversed(top_indices) if shap_vals[i] > 0]
    clean_important_features = [re.sub(r'^(num__|cat__)', '', f) for f in important_features]

    return {
        "risk_score": risk_prob,
        "risk_level": risk_level,
        "feature_importance": clean_important_features
    }


# --- CRITICAL SUBGRAPH NODES ---
def stat_alert_node(state: OverallState) -> OverallState:

    print(f"\n🚨 [CRITICAL SUBGRAPH] ALERT: Triggering Emergency Systems. Risk Score: {state['risk_score']:.2%}")

    return state

def critical_note_analysis(state: OverallState) -> Dict:

    print(f"🏥 [CRITICAL SUBGRAPH] ChatGroq parsing emergency clinical notes...")

    formatted_prompt = triage_prompt_template.format_messages(
        risk_score=f"{state.get('risk_score', 0):.2%}",
        risk_level=state.get('risk_level', 'Unknown'),
        feature_importance=", ".join(state.get('feature_importance', [])),
        vitals=str(state.get('vitals', {})),
        doctor_notes=state.get('doctor_notes', 'No clinical notes provided.'),
        patient_notes=state.get('patient_notes', 'No patient feedback provided.')
    )

    summary = structured_llm.invoke(formatted_prompt)

    return {"triage_summary": summary}

def icu_transfer_node(state: OverallState) -> Dict:

    return {"current_handler": "Emergency_ICU_Team"}

critical_builder = StateGraph(OverallState)
critical_builder.add_node("alert", stat_alert_node)
critical_builder.add_node("analyze_clinical", critical_note_analysis)
critical_builder.add_node("icu_transfer", icu_transfer_node)
critical_builder.add_edge(START, "alert")
critical_builder.add_edge("alert", "analyze_clinical")
critical_builder.add_edge("analyze_clinical", "icu_transfer")
critical_builder.add_edge("icu_transfer", END)
critical_subgraph = critical_builder.compile()

# --- STANDARD SUBGRAPH NODES ---
def queue_management_node(state: OverallState) -> OverallState:

    wait_tier = "Urgent Fast-Track" if state["risk_level"] == "Yellow" else "Routine"
    print(f"\n⏳ [STANDARD SUBGRAPH] Queue logic applied. Patient placed in {wait_tier} queue.")

    return state

def standard_note_analysis(state: OverallState) -> Dict:
    print(f"🏥 [STANDARD SUBGRAPH] ChatGroq cross-referencing routine clinical profiles...")

    formatted_prompt = triage_prompt_template.format_messages(
        risk_score=f"{state.get('risk_score', 0):.2%}",
        risk_level=state.get('risk_level', 'Unknown'),
        feature_importance=", ".join(state.get('feature_importance', [])),
        vitals=str(state.get('vitals', {})),
        doctor_notes=state.get('doctor_notes', 'No clinical notes provided.'),
        patient_notes=state.get('patient_notes', 'No patient feedback provided.')
    )

    summary = structured_llm.invoke(formatted_prompt)
    return {"triage_summary": summary}

def ward_assignment_node(state: OverallState) -> Dict:
    summary = state.get("triage_summary")
    
    # Check if the LLM caught an emergency that the ML model missed
    if summary and (summary.recommended_action == "ICU" or summary.triage_urgency == "High"):
        print("🚨 SAFETY OVERRIDE DETECTED: Upgrading handler to Emergency Team due to LLM clinical findings.")
        return {
            "current_handler": "Emergency_ICU_Team",
            "risk_level": "Red" # Correct the visual classification for the dashboard
        }
        
    return {"current_handler": "Standard_Ward_Team"}

standard_builder = StateGraph(OverallState)
standard_builder.add_node("queue", queue_management_node)
standard_builder.add_node("analyze_feedback", standard_note_analysis)
standard_builder.add_node("ward_assignment", ward_assignment_node)
standard_builder.add_edge(START, "queue")
standard_builder.add_edge("queue", "analyze_feedback")
standard_builder.add_edge("analyze_feedback", "ward_assignment")
standard_builder.add_edge("ward_assignment", END)
standard_subgraph = standard_builder.compile()


# Database Log & Router 
def database_update_node(state: OverallState) -> OverallState:
    """Writes the comprehensive final LangGraph state into Postgres Database"""
    print(f"\n💾 --- Logging Workflow State ---")
    
    if not db_engine:
        print("⚠️ Database engine not available. Skipping insertion.")
        return state

    v = state["vitals"]
    summary = state.get("triage_summary")
    
    try:
        # Using context manager 'begin()' handles automatic commit/rollback configurations
        with db_engine.begin() as conn:
            result = conn.execute(
                sqlalchemy.text("SELECT 1 FROM triage_sessions WHERE patient_id = :patient_id"),
                {"patient_id": state.get("patient_id")}
            ).fetchone()
            
            params = {
                "patient_id": state.get("patient_id"),
                "age_at_triage": v["Age"], 
                "heart_rate_bpm": v["HeartRate_bpm"], 
                "resp_rate_bpm": v["RespRate_bpm"],
                "systolic_bp": v["SystolicBP"], 
                "diastolic_bp": v["DiastolicBP"], 
                "temperature_c": v["Temperature_C"],
                "spo2_percent": v["SpO2_percent"], 
                "fasting_glucose_mg_dl": v["FastingGlucose_mg_dL"],
                "total_cholesterol_mg_dl": v["TotalCholesterol_mg_dL"], 
                "weight_kg": v["Weight_kg"],
                "height_cm": v["Height_cm"], 
                "bmi": v["BMI"], 
                "pain_level": v["Pain_Level_0_to_10"],
                "mental_status": v["Mental_Status"], 
                "symptom_duration_days": v["Symptom_Duration_Days"],
                "patient_notes": state.get("patient_notes"), 
                "doctor_notes": state.get("doctor_notes"),
                "risk_score": state.get("risk_score"), 
                "risk_level": state.get("risk_level"),
                "feature_importance": state.get("feature_importance"),
                "triage_urgency": summary.triage_urgency if summary else None,
                "primary_system_affected": summary.primary_system_affected if summary else None,
                "symptom_progression": summary.symptom_progression if summary else None,
                "recommended_action": summary.recommended_action if summary else None,
                "distress_level": summary.distress_level if summary else None,
                "potential_diagnosis_category": summary.potential_diagnosis_category if summary else None
            }

            if result:
                conn.execute(sqlalchemy.text("""
                    UPDATE triage_sessions SET
                        age_at_triage = :age_at_triage, heart_rate_bpm = :heart_rate_bpm, resp_rate_bpm = :resp_rate_bpm, 
                        systolic_bp = :systolic_bp, diastolic_bp = :diastolic_bp, temperature_c = :temperature_c, 
                        spo2_percent = :spo2_percent, fasting_glucose_mg_dl = :fasting_glucose_mg_dl,
                        total_cholesterol_mg_dl = :total_cholesterol_mg_dl, weight_kg = :weight_kg, height_cm = :height_cm, 
                        bmi = :bmi, pain_level = :pain_level, mental_status = :mental_status, 
                        symptom_duration_days = :symptom_duration_days, patient_notes = :patient_notes, doctor_notes = :doctor_notes, 
                        risk_score = :risk_score, risk_level = :risk_level, feature_importance = :feature_importance, 
                        triage_urgency = :triage_urgency, primary_system_affected = :primary_system_affected,
                        symptom_progression = :symptom_progression, recommended_action = :recommended_action, 
                        distress_level = :distress_level, potential_diagnosis_category = :potential_diagnosis_category, 
                        status = 'Completed', updated_at = CURRENT_TIMESTAMP
                    WHERE patient_id = :patient_id;
                """), params)
            else:
                conn.execute(sqlalchemy.text("""
                    INSERT INTO triage_sessions (
                        patient_id, age_at_triage, heart_rate_bpm, resp_rate_bpm, systolic_bp,
                        diastolic_bp, temperature_c, spo2_percent, fasting_glucose_mg_dl, total_cholesterol_mg_dl,
                        weight_kg, height_cm, bmi, pain_level, mental_status, symptom_duration_days,
                        patient_notes, doctor_notes, risk_score, risk_level, feature_importance,
                        triage_urgency, primary_system_affected, symptom_progression, recommended_action,
                        distress_level, potential_diagnosis_category, status
                    ) VALUES (
                        :patient_id, :age_at_triage, :heart_rate_bpm, :resp_rate_bpm, :systolic_bp,
                        :diastolic_bp, :temperature_c, :spo2_percent, :fasting_glucose_mg_dl, :total_cholesterol_mg_dl,
                        :weight_kg, :height_cm, :bmi, :pain_level, :mental_status, :symptom_duration_days,
                        :patient_notes, :doctor_notes, :risk_score, :risk_level, :feature_importance,
                        :triage_urgency, :primary_system_affected, :symptom_progression, :recommended_action,
                        :distress_level, :potential_diagnosis_category, 'Completed'
                    );
                """), params)
        print("💾 Postgres database transaction completed successfully.")
    except Exception as e:
        print(f"❌ Postgres Insertion Error: {e}")
    return state

def route_by_risk(state: OverallState) -> str:
    if state["risk_level"] == "Red":
        return "critical"
    return "standard"

workflow = StateGraph(OverallState,input_schema=PatientInput)

workflow.add_node("validate", validate_vitals_node)
workflow.add_node("recount_vitals", recount_vitals_node)
workflow.add_node("predict", predictive_inference_node)
workflow.add_node("db_update", database_update_node)
workflow.add_node("critical_path", critical_subgraph)
workflow.add_node("standard_path", standard_subgraph)

workflow.add_edge(START, "validate")
workflow.add_conditional_edges("validate", should_recount, {"recount": "recount_vitals", "proceed": "predict"})
workflow.add_edge("recount_vitals", "validate")
workflow.add_conditional_edges("predict", route_by_risk, {"critical": "critical_path", "standard": "standard_path"})
workflow.add_edge("critical_path", "db_update")
workflow.add_edge("standard_path", "db_update")
workflow.add_edge("db_update", END)

app = workflow.compile()