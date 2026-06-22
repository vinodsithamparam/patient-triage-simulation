import os
import uuid
import streamlit as st
import pandas as pd
import sqlalchemy
import pg8000
from langgraph.pregel.remote import RemoteGraph

st.set_page_config(layout="wide", page_title="Clinical Triage Center")
st.title("🏥 Patient Triage Control Center")

# --- Infrastructure Environment Variables ---
GRAPH_URL = os.environ.get("CLOUD_RUN_GRAPH_URL")
GRAPH_NAME = os.environ.get("GRAPH_NAME", "triage")
DATABASE_URL = os.environ.get("DATABASE_URL")

def init_remote_graph_client(url, graph_name):
    """Initializes the LangGraph SDK RemoteGraph reference targeting the local network container."""
    if not url:
        return None
    return RemoteGraph(graph_name, url=url)

def get_db_engine() -> sqlalchemy.engine.base.Engine:
    """Initializes a standard connection pool targeting the local PostgreSQL container network."""
    if DATABASE_URL:
        return sqlalchemy.create_engine(DATABASE_URL)
    return None

def style_triage_cells(val):
    """Applies conditional CSS styling matching clinical risk tiers."""
    if val in ['Red', 'High']:
        return 'background-color: #fce8e6; color: #a51d24; font-weight: bold; border-radius: 4px;'
    elif val in ['Yellow', 'Medium']:
        return 'background-color: #fef7e0; color: #b06000; font-weight: bold; border-radius: 4px;'
    elif val in ['Green', 'Low']:
        return 'background-color: #e6f4ea; color: #137333; font-weight: bold; border-radius: 4px;'
    return ''

remote_graph = init_remote_graph_client(GRAPH_URL, GRAPH_NAME)
db_engine = get_db_engine()

# --- UI Tabs Interface Setup ---
tab1, tab2 = st.tabs(["📋 Patient Registration Form", "📊 Clinical Triage Dashboard"])

with tab1:
    st.header("Emergency Intake Registration")
    
    if not remote_graph:
        st.warning("⚠️ CLOUD_RUN_GRAPH_URL configuration missing. Operating in simulation mode.")
        
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            p_id = st.text_input("Patient Reference ID", "991002026411")
            age = st.number_input("Age", 1, 110, 54)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            mental = st.selectbox("Mental Status Assessment", ["Alert", "Confused", "Lethargic", "Unresponsive"])
        with c2:
            hr = st.slider("Heart Rate (bpm)", 40, 220, 82)
            rr = st.slider("Respiratory Rate (bpm)", 8, 45, 18)
            sys_bp = st.number_input("Systolic Blood Pressure", 60, 230, 135)
            dia_bp = st.number_input("Diastolic Blood Pressure", 40, 140, 88)
        with c3:
            spo2 = st.slider("Oxygen Saturation - SpO2 (%)", 50, 100, 96)
            temp = st.number_input("Temperature (°C)", 34.0, 43.0, 37.2, step=0.1)
            pain = st.slider("Pain Threshold Score (0-10)", 0, 10, 6)
            duration = st.number_input("Onset Duration (Days)", 0, 30, 3)

        st.subheader("Clinical Narratives")
        doc_notes = st.text_area("Physician Initial Clinical Notes", "Patient exhibits acute non-radiating chest pressure. S4 gallop audible.")
        pat_notes = st.text_area("Patient / Family Feedback Notes", "Feels like a heavy elephant is sitting on my chest.")

        submit = st.form_submit_button("⚡ Run Graph Engine Execution Pipeline")

        if submit and remote_graph:
            session_id = str(uuid.uuid4())
            
            if spo2 > 100 or spo2 < 20 or hr > 250 or hr < 0:
                st.error("❌ Form validation failed: Critical metrics are out of logical diagnostic bounds.")
            else:
                payload_state = {
                    "patient_id": int(p_id) if p_id.isdigit() else 9921,
                    "session_id": session_id,
                    "patient_notes": pat_notes,
                    "doctor_notes": doc_notes,
                    "vitals": {
                        "Age": int(age), "Gender": gender, "HeartRate_bpm": int(hr), "RespRate_bpm": int(rr),
                        "SystolicBP": int(sys_bp), "DiastolicBP": int(dia_bp), "Temperature_C": float(temp),
                        "SpO2_percent": int(spo2), "FastingGlucose_mg_dL": 90, "TotalCholesterol_mg_dL": 180,
                        "Weight_kg": 80.0, "Height_cm": 178.0, "BMI": 25.2, "Pain_Level_0_to_10": int(pain),
                        "Mental_Status": mental, "Symptom_Duration_Days": int(duration)
                    },
                    "risk_score": 0.0, "risk_level": "Green", "feature_importance": [],
                    "validation_errors": [], "invalid_keys": [], "current_handler": "Intake"
                }
                
                with st.spinner("Invoking Local LangGraph Engine Framework..."):
                    try:
                        final_state = remote_graph.invoke(payload_state)
                        print(final_state)
                        summary = final_state.get("triage_summary")

                        
                        st.success(f"Processing Complete! Active Session ID: {session_id}")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Risk Classification", final_state.get("risk_level", "Unknown"), 
                                  delta=f"Score: {final_state.get('risk_score', 0.0):.1%}")
                        m2.metric("Pipeline Stage Assignment", final_state.get("current_handler", "N/A"))
                        m3.write(f"**Primary Key Drivers:** {', '.join(final_state.get('feature_importance', []))}")
                        
                        if summary:
                            st.subheader("Parser Object Details")
                            st.json(summary)
                            
                    except Exception as e:
                        st.error(f"Local LangGraph Runtime Exception: {e}")

with tab2:
    st.header("📊 Clinical Metrics & Queue Dashboard")
    
    col_refresh, col_spacer = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh Live Dashboard"):
            st.rerun()

    if db_engine:
        try:
            with db_engine.connect() as conn:
                query = """
                    SELECT 
                        patient_id, age_at_triage, heart_rate_bpm, resp_rate_bpm, 
                        systolic_bp, diastolic_bp, temperature_c, spo2_percent, 
                        fasting_glucose_mg_dl, total_cholesterol_mg_dl, weight_kg, 
                        height_cm, bmi, pain_level, mental_status, symptom_duration_days, 
                        patient_notes, doctor_notes, risk_score, risk_level, 
                        triage_urgency, primary_system_affected, symptom_progression, 
                        recommended_action, distress_level, potential_diagnosis_category, 
                        status, created_at, updated_at 
                    FROM triage_sessions 
                    ORDER BY updated_at DESC
                """
                df = pd.read_sql(query, conn)
                if not df.empty:
                    df['patient_id'] = df['patient_id'].astype('Int64')
                
            if not df.empty:
                # --- SECTION 1: Key Operational Metrics ---
                st.subheader("📈 Operational Summary")
                m1, m2, m3, m4 = st.columns(4)
                
                total_patients = len(df)
                pending_triage = len(df[df['status'].str.lower().str.strip() != 'completed']) if 'status' in df.columns else 0
                avg_risk = df['risk_score'].astype(float).mean() if 'risk_score' in df.columns else 0.0
                critical_count = len(df[df['risk_level'].str.lower().str.strip() == 'red']) if 'risk_level' in df.columns else 0

                m1.metric("Total Registry Count", f"{total_patients} Patients")
                m2.metric("Active Dynamic Queues", f"{pending_triage} Pending")
                m3.metric("Mean Population Risk", f"{avg_risk:.1%}")
                m4.metric("Critical Red Status Alert", f"{critical_count} Urgent Cases")
                
                st.markdown("---")

                # --- SECTION 2: Detailed Queue Stream Grid View ---
                st.subheader("📋 PostgreSQL Local Queue Registry Stream")

                # Formatted data grid columns for UI visibility
                display_cols = [
                    "patient_id", "age_at_triage", "triage_urgency", "risk_level", 
                    "risk_score", "primary_system_affected", "recommended_action", 
                    "status", "updated_at"
                ]
                available_cols = [c for c in display_cols if c in df.columns]

                # Target both triage indicators for highlighting if they are present in the dataset
                target_style_cols = [c for c in ['triage_urgency', 'risk_level'] if c in available_cols]

                if target_style_cols:
                    # Apply the mapping function exclusively to our target column subsets
                    styled_df = df[available_cols].style.map(style_triage_cells, subset=target_style_cols)
                    
                    # Render the styled dataframe matrix
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(df[available_cols], use_container_width=True)
                
                # --- SECTION 3: Expanded Patient Record Inspector ---
                st.markdown("---")
                st.subheader("🔍 Individual Case File Deep-Dive")
                selected_patient = st.selectbox(
                    "Select Patient Reference ID to view clinical narratives:", 
                    options=df['patient_id'].unique(),
                    format_func=lambda x: f"{int(x)}" if pd.notnull(x) else "N/A"
                )
                
                if selected_patient:
                    patient_row = df[df['patient_id'] == selected_patient].iloc[0]
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.markdown(f"**Patient Notes:**")
                        st.write(patient_row.get('patient_notes', 'N/A'))
                        st.markdown(f"**Potential Diagnosis Category:**")
                        st.write(patient_row.get('potential_diagnosis_category', 'N/A'))
                    with col_p2:
                        st.markdown(f"**Physician Clinical Notes:**")
                        st.write(patient_row.get('doctor_notes', 'N/A'))
                        st.markdown(f"**Feature Significance Framework Markers:**")
                        st.write(f"Distress Level Rating: {patient_row.get('distress_level', 'N/A')} | Progression: {patient_row.get('symptom_progression', 'N/A')}")

            else:
                st.info("No active patient sessions found in your local database instances.")
                
        except Exception as e:
            st.error(f"Failed to fetch records from the local PostgreSQL engine: {e}")
    else:
        st.error("No valid database engine available. Check your container network environment variables.")