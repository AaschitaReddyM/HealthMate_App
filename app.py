# app.py - HealthMate Frontend with Insulin Tracker & SHAP Explanation (Text Summary to Chatbot)

import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont # Import Pillow modules
import joblib  # For loading model/preprocessors locally
import shap    # For SHAP explanations
import matplotlib.pyplot as plt # For displaying SHAP plot
import traceback # For detailed error logging
import sys # To check python version if needed

# --- Print Versions (Helpful for Debugging Environments) ---
print(f"Streamlit version: {st.__version__}")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"Joblib version: {joblib.__version__}")
print(f"Pillow version: {Image.__version__}")
print(f"SHAP version: {shap.__version__}")

# --- Page Configuration ---
st.set_page_config(
    page_title="HealthMate App",
    page_icon="🩺",
    layout="wide"
)

# --- Configure Backend API URL ---
BACKEND_URL = os.environ.get("BACKEND_API_URL", "https://us-central1-healthmate-app-457518.cloudfunctions.net/healthmate-predict") # Use your deployed URL
api_url = BACKEND_URL
print(f"Using Backend URL: {api_url}")

# --- Configure Generative AI Model (Gemini) ---
try:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")
    if not gemini_api_key:
        st.warning("Gemini API Key not found in secrets. Chatbot will be disabled.", icon="🔑")
        gemini_model = None
    else:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17")
        print("Gemini model configured.")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Chatbot will be disabled.", icon="🚨")
    gemini_model = None

# --- Load Saved Model and Preprocessing Objects for XAI ---
MODEL_DIR_APP = os.path.dirname(os.path.abspath(__file__))

# !!! IMPORTANT: Update MODEL_FILENAME_APP to match the file YOU saved !!!
MODEL_FILENAME_APP = 'CatBoost_No Imbalance Handling_best_model.joblib' # EXAMPLE - CHANGE THIS
SCALER_FILENAME_APP = 'scaler.joblib'
IMPUTER_FILENAME_APP = 'imputer.joblib'

@st.cache_resource
def load_prediction_artifacts(model_file, scaler_file, imputer_file):
    """Loads the model, scaler, and imputer required for frontend explanations."""
    model, scaler, imputer = None, None, None
    print(f"Attempting to load artifacts for XAI from: {MODEL_DIR_APP}")
    all_loaded = True
    try:
        model_path = os.path.join(MODEL_DIR_APP, model_file)
        scaler_path = os.path.join(MODEL_DIR_APP, scaler_file)
        imputer_path = os.path.join(MODEL_DIR_APP, imputer_file)

        if os.path.exists(model_path): model = joblib.load(model_path); print(f"XAI: Model loaded successfully from {model_path}.")
        else: print(f"XAI Error: Model file not found at {model_path}"); all_loaded = False

        if os.path.exists(scaler_path): scaler = joblib.load(scaler_path); print(f"XAI: Scaler loaded successfully from {scaler_path}.")
        else: print(f"XAI Error: Scaler file not found at {scaler_path}"); all_loaded = False

        if os.path.exists(imputer_path): imputer = joblib.load(imputer_path); print(f"XAI: Imputer loaded successfully from {imputer_path}.")
        else: print(f"XAI Error: Imputer file not found at {imputer_path}"); all_loaded = False

        if not all_loaded:
             st.error("One or more required files (model, scaler, imputer) not found. Explanations disabled.", icon="🚨")
             return None, None, None
        return model, scaler, imputer
    except Exception as e:
        print(f"XAI Error loading artifacts in frontend: {e}"); st.error(f"Could not load components for explanation: {e}", icon="⚠️"); traceback.print_exc(); return None, None, None

loaded_model, loaded_scaler, loaded_imputer = load_prediction_artifacts(MODEL_FILENAME_APP, SCALER_FILENAME_APP, IMPUTER_FILENAME_APP)
xai_artifacts_loaded = all([loaded_model, loaded_scaler, loaded_imputer])
if not xai_artifacts_loaded: st.sidebar.warning("Explanations disabled: Failed to load model/scaler/imputer.", icon="⚠️")

# --- Define Features (Must match training order) ---
FEATURES = ['age', 'gender', 'ethnicity', 'glucose', 'hba1c', 'insulin', 'weight']

# --- Define Mapping Dictionaries and Binning Functions (Must match main.py/training) ---
INSULIN_MAPPING = {'no': 0, 'steady': 1, 'down': 2, 'up': 3}
def map_glucose_bins(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return np.nan
    try: value = float(value)
    except (ValueError, TypeError): return np.nan
    if 200 <= value < 300: return 1
    elif value >= 300: return 3
    elif value == 0: return 0
    else: return 2
def map_hba1c_bins(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return np.nan
    try: value = float(value)
    except (ValueError, TypeError): return np.nan
    if 7 <= value < 8: return 1
    elif value >= 8: return 3
    elif value == 0: return 0
    else: return 2
ETHNICITY_MAPPING = {'caucasian': 3.0, 'africanamerican': 4.0, 'asian': 5.0, 'hispanic': 2.0, 'other': 5.0}
GENDER_MAPPING = {'male': 1, 'female': 2, 'other': np.nan}
def apply_mapping(value, mapping):
    if value is None or (isinstance(value, (float, int)) and np.isnan(value)): return np.nan
    value_str = str(value).strip().lower()
    mapping_lower = {k.lower(): v for k, v in mapping.items()}
    return mapping_lower.get(value_str, np.nan)

# --- Preprocessing Function for SHAP ---
def preprocess_data_for_shap(input_data_dict, features_list, scaler, imputer):
    if not xai_artifacts_loaded: print("Preprocessing skipped: Scaler or Imputer not loaded."); st.error("Preprocessing failed: Required components not loaded.", icon="⚠️"); return None, None
    try:
        print("Starting frontend preprocessing for SHAP...")
        input_df_data = {feature: input_data_dict.get(feature) for feature in features_list}
        input_df = pd.DataFrame([input_df_data], columns=features_list)
        processed_df = input_df.copy()
        # Apply mappings/binning
        if 'gender' in processed_df.columns: processed_df['gender'] = processed_df['gender'].apply(lambda x: apply_mapping(x, GENDER_MAPPING))
        if 'ethnicity' in processed_df.columns: processed_df['ethnicity'] = processed_df['ethnicity'].apply(lambda x: apply_mapping(x, ETHNICITY_MAPPING))
        if 'insulin' in processed_df.columns: processed_df['insulin'] = processed_df['insulin'].apply(lambda x: apply_mapping(x, INSULIN_MAPPING))
        if 'glucose' in processed_df.columns: processed_df['glucose'] = processed_df['glucose'].apply(map_glucose_bins)
        if 'hba1c' in processed_df.columns: processed_df['hba1c'] = processed_df['hba1c'].apply(map_hba1c_bins)
        # Coerce and impute/scale
        for col in features_list:
            if col in processed_df.columns: processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            else: print(f"Warning: Feature '{col}' not found during numeric coercion."); processed_df[col] = np.nan
        processed_df = processed_df[features_list]
        imputed_array = loaded_imputer.transform(processed_df)
        imputed_df = pd.DataFrame(imputed_array, columns=features_list)
        scaled_data_np = loaded_scaler.transform(imputed_df)
        print("Frontend preprocessing for SHAP successful.")
        return scaled_data_np, imputed_df
    except Exception as e: st.error(f"Error during frontend preprocessing for SHAP: {e}", icon="⚠️"); print(f"Error during frontend preprocessing: {e}"); traceback.print_exc(); return None, None

def display_digital_twin_section(simulations_data):
    """
    Uses Streamlit components to display the Digital Twin simulation results.
    Handles specific text for Insulin change simulation.
    """
    st.subheader("🚀 Digital Twin: Potential Goals for Next 30 Days")
    st.caption("Simulating the potential impact of specific changes on your risk score.")

    if not simulations_data:
        st.info("No specific impact simulations are available based on your current profile.", icon="ℹ️")
        return

    score_display_tolerance = 0.1 # Minimum % change in score to show numerically

    # Define paths to local icon images
    icons_dir = os.path.join(os.path.dirname(__file__), "icons")
    icon_paths = {
        "glucose": os.path.join(icons_dir, "glucose-meter.png"),
        "hba1c": os.path.join(icons_dir, "document.png"), # Suggestion
        "insulin": os.path.join(icons_dir, "insulin.png"), # Suggestion for insulin icon
        # "weight": os.path.join(icons_dir, "weight-scale.png"), # Removed weight
        "default": os.path.join(icons_dir, "default-icon.png")
    }

    for sim in simulations_data:
        try:
            factor = sim.get('factor_name', 'Unknown Factor')
            factor_change = sim.get('percent_change_factor') # Can be None for insulin
            score_change = sim.get('percent_change_score', 0.0)

            # --- Determine text for factor change ---
            # ... (keep the existing logic for factor_text) ...
            factor_text = ""
            if factor == "insulin":
                factor_text = f"change **{factor}** status to **'No'**"
            elif factor_change is not None:
                if factor_change < 0:
                    factor_text = f"reduce **{factor}** by **{abs(factor_change):.0f}%**"
                elif factor_change > 0:
                    factor_text = f"increase **{factor}** by **{factor_change:.0f}%**"
                else:
                    continue
            else:
                print(f"Warning: Skipping display for factor '{factor}' due to missing change info.")
                continue

            # Determine text and styling for score impact (logic remains the same)
            # ... (keep the existing logic for score_impact_text and color) ...
            score_impact_text = ""
            color = "#555555" # Default/grey
            if abs(score_change) <= score_display_tolerance:
                score_impact_text = "have **no significant impact** on your score."
            elif score_change < 0: # Assumes LOWER score is better
                score_impact_text = f"potentially **DECREASE** your score by **{abs(score_change):.0f}%**."
                color = "#28a745" # Green
            else: # Assumes LOWER score is better
                score_impact_text = f"potentially **INCREASE** your score by **{score_change:.0f}%**."
                color = "#dc3545" # Red


            # Select icon (logic remains the same)
            icon_key = factor.lower()
            icon_file = icon_paths.get(icon_key, icon_paths["default"])

            # === Display using columns for layout (with adjustments) ===
            # Try a different ratio, giving less relative space to the icon column
            col1, col2 = st.columns([1, 12])  # <-- ADJUSTED RATIO (e.g., from [1, 9])

            with col1:
                if os.path.exists(icon_file):
                    # Increase the icon width
                    st.image(icon_file, width=40) # <-- ADJUSTED WIDTH (e.g., from 30)
                else:
                    st.write("🔹") # Fallback

            with col2:
                # Display text (no change here)
                st.markdown(f"""
                If you {factor_text} over the next 30 days, it could
                <span style='color:{color}; font-weight:bold;'>{score_impact_text}</span>
                """, unsafe_allow_html=True)
            # === End display section ===

        except Exception as display_e:
             st.warning(f"Could not display simulation for {factor}: {display_e}", icon="⚠️")

    st.caption("ℹ️ Remember, these are model-based simulations. Consult your doctor.")
    st.divider()

# --- Initialize SHAP Explainer (Cached) ---
@st.cache_resource
def get_shap_explainer(_model): # Use underscore prefix for unhashable args
    """Initializes and returns a SHAP Explainer suitable for the model."""
    if _model is None: print("SHAP Explainer cannot be initialized: Model object is None."); return None
    model_type_str = str(type(_model)).lower(); print(f"Initializing SHAP explainer for model type: {model_type_str}")
    if 'xgboost' in model_type_str or 'lightgbm' in model_type_str or 'catboost' in model_type_str:
        try:
            if hasattr(_model, 'is_fitted') and not _model.is_fitted(): print("Warning: Model provided to SHAP explainer is not fitted.")
            explainer = shap.TreeExplainer(_model)
            try: _ = explainer.expected_value; print(f"SHAP TreeExplainer initialized. Expected value type: {type(explainer.expected_value)}")
            except Exception as ev_e: print(f"Could not pre-access expected_value: {ev_e}")
            return explainer
        except Exception as e: st.error(f"Failed to initialize SHAP TreeExplainer: {e}", icon="⚠️"); print(f"Failed to initialize SHAP TreeExplainer: {e}"); traceback.print_exc(); return None
    else: print(f"Model type {type(_model)} might require non-TreeExplainer."); st.warning(f"Model type {type(_model)} may not be supported by TreeExplainer.", icon="⚠️"); return None

shap_explainer = get_shap_explainer(loaded_model) if xai_artifacts_loaded else None

# --- Helper functions for state changes ---
def go_to_intro(): st.session_state.page = 'intro'
def go_to_risk_prediction():
    st.session_state.page = 'risk_prediction'
    if 'last_risk_score' in st.session_state: del st.session_state.last_risk_score
    if 'last_prediction_data' in st.session_state: del st.session_state.last_prediction_data
    if 'last_shap_values' in st.session_state: del st.session_state.last_shap_values # Clear old SHAP if needed
    if 'plot_summary_text' in st.session_state: del st.session_state.plot_summary_text # Clear old summary

def go_to_chatbot_tool_with_context(context_type="general", plot_summary=None): # Accept plot_summary
    """Navigates to chatbot and sets context type and plot summary in session state."""
    st.session_state.chatbot_context = context_type # e.g., 'plot_explanation'
    st.session_state.plot_summary_text = plot_summary # Store the summary text
    print(f"Navigating to chatbot with context: {context_type}, Summary provided: {plot_summary is not None}")
    go_to_chatbot_tool() # Call the original navigation function

def go_to_chatbot_tool():
    if gemini_model is None:
        st.warning("AI Assistant Chatbot is not available.", icon="🚨")
        return
    st.session_state.page = 'chatbot_tool'
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = []

    if 'gemini_chat_session' not in st.session_state or st.session_state.gemini_chat_session is None:
        try:
            st.session_state.gemini_chat_session = gemini_model.start_chat(history=[])
            print("Gemini chat session started.")

            initial_message = "Hi! I'm the HealthMate Assistant. How can I help?"
            context_type = st.session_state.pop('chatbot_context', 'general')
            plot_summary = st.session_state.pop('plot_summary_text', None) # Get and remove plot summary

            initial_context_parts = []
            if context_type == "plot_explanation" and plot_summary:
                # --- NEW, VERY FOCUSED INSTRUCTIONS FOR PLOT EXPLANATION ---
                initial_context_parts.append(
                "SYSTEM INSTRUCTION: Your task is to analyze the provided SHAP summary text below.\n\n"
                f"SHAP SUMMARY:\n---\n{plot_summary}\n---\n\n"
                "TASK: Identify and list the **top 2 features** from the summary that **increased** the risk score the most (had the most positive contribution value). For each, state the feature name and its contribution value.\n"
                "Then, identify and list the **top 2 features** from the summary that **decreased** the risk score the most (had the most negative contribution value). For each, state the feature name and its contribution value.\n"
                "**Only provide these lists.** Do not add general interpretations or lifestyle advice. Finish with the standard medical disclaimer."
                )
                # Keep the initial user-facing message the same or similar
                initial_message = "I have the summary of feature contributions. Ask me to list the top factors increasing or decreasing the score."
                print("DEBUG: Using SIMPLIFIED PLOT EXPLANATION context for Gemini.")
                # --- END OF SIMPLIFIED INSTRUCTIONS ---
            elif 'last_risk_score' in st.session_state:
                 # --- Existing General Score Context Logic ---
                 # ... (keep this part as it was) ...
                 print("DEBUG: Using GENERAL SCORE context for Gemini.")

            if initial_context_parts:
                 full_initial_context = "\n".join(initial_context_parts)
                 # --- ADD PRINT STATEMENT HERE ---
                 print("=" * 40)
                 print("DEBUG: FINAL PROMPT BEING SENT TO GEMINI:")
                 print(full_initial_context)
                 print("=" * 40)
                 # --- END PRINT STATEMENT ---
                 try:
                     st.session_state.gemini_chat_session.send_message(full_initial_context)
                     print(f"Sent initial context ({context_type}) to Gemini.")
                 except Exception as context_e: print(f"Could not send initial context to chatbot: {context_e}")

            if not st.session_state.chatbot_messages: # Add starting message only if history is empty
                 st.session_state.chatbot_messages.append({"role": "assistant", "content": initial_message})

        except Exception as start_e:
             st.error(f"Could not start Gemini chat session: {start_e}", icon="🚨")
             st.session_state.gemini_chat_session = None
             st.session_state.pop('chatbot_context', None); st.session_state.pop('plot_summary_text', None)
    else: # Clear flags if session already existed but we navigated with context again
         st.session_state.pop('chatbot_context', None)
         st.session_state.pop('plot_summary_text', None)

def go_to_post_diagnosis_care():
    st.session_state.page = 'post_diagnosis_care'
    if 'last_injection_site' not in st.session_state: st.session_state.last_injection_site = None
    if 'suggested_next_site' not in st.session_state: st.session_state.suggested_next_site = None

# --- Manage State with st.session_state ---
if 'page' not in st.session_state: go_to_intro()

# --- Define Page/View Functions ---

def intro_page():
    """Renders the main introduction and navigation page."""
    st.title("🩺 Welcome to HealthMate")
    st.markdown("Your personal health companion. Select a tool below.")

    # --- NEW SECTION FOR GENERAL INFO LINKS ---
    st.divider() # Optional separator
    st.subheader("Learn More About Diabetes")
    st.markdown(
        """
        Here are some reliable resources for general information about diabetes:

        * **[Centers for Disease Control and Prevention (CDC) - Diabetes Basics](https://www.cdc.gov/diabetes/basics/index.html)**
            * *Provides comprehensive information on types, symptoms, prevention, and management.*
        * **[National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)](https://www.niddk.nih.gov/health-information/diabetes)**
            * *Offers detailed health information and research updates from the National Institutes of Health (NIH).*
        * **[American Diabetes Association (ADA)](https://diabetes.org/)**
            * *A leading advocacy organization with extensive resources, recipes, news, and support information.*

        """
    )
    st.caption("These external links provide general educational content. They are not a substitute for professional medical advice from your doctor.")
    # --- END NEW SECTION ---
    
    st.divider(); st.subheader("Available Tools:")
    c1, c2, c3 = st.columns(3)
    with c1: st.button("📊 Diabetes Risk Prediction", on_click=go_to_risk_prediction, use_container_width=True)
    with c2: st.button("💬 HealthMate AI Assistant", on_click=go_to_chatbot_tool, disabled=(gemini_model is None), use_container_width=True);
    if gemini_model is None: st.caption("Chatbot disabled.")
    with c3: st.button("💉 Insulin Site Tracker", on_click=go_to_post_diagnosis_care, use_container_width=True)
    st.divider(); st.caption("Disclaimer: HealthMate provides estimates and general info. Always consult a healthcare professional.")

# Make sure necessary imports like streamlit, requests, os, math, etc.
# and variables like FEATURES, api_url, gemini_model, xai_artifacts_loaded,
# shap_explainer, loaded_model, loaded_scaler, loaded_imputer are defined
# Also ensure helper functions like go_to_intro, go_to_chatbot_tool_with_context,
# preprocess_data_for_shap, and display_digital_twin_section are defined.

def risk_prediction_tool():
    """Renders the Diabetes Risk Prediction tool interface."""
    st.title("📊 Diabetes Risk Score Prediction")
    st.markdown("Enter patient information to predict diabetes likelihood.")
    st.divider()
    st.button("← Back to Tools", on_click=go_to_intro) # Assuming go_to_intro is defined
    st.divider()

    with st.form(key="prediction_form"):
        st.subheader("Patient Information:")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 58, 1, help="Years")
            glucose = st.number_input("Glucose (mg/dL)", 0.0, value=95.0, step=0.1, format="%.1f", help="Fasting glucose")
            weight = st.number_input("Weight (kg)", 0.0, value=75.0, step=0.1, format="%.1f")
        with col2:
            gender = st.selectbox("Gender", ['Male', 'Female', 'Other'], index=0)
            ethnicity = st.selectbox("Ethnicity", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'], index=0)
            hba1c = st.number_input("HbA1c (%)", 0.0, 20.0, 5.7, 0.1, format="%.1f")
        insulin_status = st.selectbox("Insulin Status", ['No', 'Steady', 'Down', 'Up'], index=0, help="'No' = not using insulin")
        submit_button = st.form_submit_button(label="Predict Risk Score", use_container_width=True)
    st.divider()

    if submit_button:
        # Clear previous results from session state
        if 'last_risk_score' in st.session_state: del st.session_state.last_risk_score
        if 'last_prediction_data' in st.session_state: del st.session_state.last_prediction_data
        if 'last_shap_values' in st.session_state: del st.session_state.last_shap_values
        if 'plot_summary_text' in st.session_state: del st.session_state.plot_summary_text

        # Prepare data for API
        input_data = {"age": age, "gender": gender, "ethnicity": ethnicity, "glucose": glucose, "hba1c": hba1c, "insulin": insulin_status, "weight": weight}
        api_payload = {feature: input_data.get(feature) for feature in FEATURES} # Assuming FEATURES list is defined globally

        st.info(f"Sending data to backend: {api_url}", icon="☁️") # Assuming api_url is defined globally
        print(f"API Payload Sent: {api_payload}")

        try:
            # Make API Call
            with st.spinner("Calculating risk score and simulations..."):
                response = requests.post(api_url, json=api_payload, timeout=45) # Increased timeout slightly
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            print(f"API Response Received: {result}")

            # Process Successful Response
            if "risk_score" in result and isinstance(result["risk_score"], (int, float)):
                risk_score = result["risk_score"]
                st.subheader("Prediction Result:")
                st.metric(label="Predicted Diabetes Risk Score", value=f"{risk_score:.4f}")

                # Display Risk Level
                low_thresh, mod_thresh = 0.3, 0.6
                if risk_score < low_thresh:
                    st.success("Risk level: **Low**.", icon="✅")
                    # st.balloons() # Optional celebration
                elif low_thresh <= risk_score < mod_thresh:
                    st.info("Risk level: **Moderate**.", icon="⚠️")
                else:
                    st.warning("Risk level: **High**. Consult a healthcare professional.", icon="🚨")

                # Store results in session state
                st.session_state.last_risk_score = risk_score
                st.session_state.last_prediction_data = input_data

                # --- Display General Chatbot Button ---
                st.button("💬 Discuss Risk Score with Assistant",
                          key="ask_chatbot_about_score",
                          on_click=go_to_chatbot_tool_with_context, # Assuming this function is defined
                          args=("general_score", None), # Pass context type, no plot summary
                          use_container_width=True,
                          disabled=(gemini_model is None)) # Assuming gemini_model check is appropriate

                st.divider() # Separator before Digital Twin

                # --- Display Digital Twin Section (NEW) ---
                # Assumes backend returns 'digital_twin_simulations' list in JSON
                if "digital_twin_simulations" in result and isinstance(result["digital_twin_simulations"], list):
                    display_digital_twin_section(result["digital_twin_simulations"]) # Call the display function
                else:
                    st.info("Digital twin simulation data not available in the response.", icon="ℹ️")
                    print("Backend response did not contain 'digital_twin_simulations' list.")
                # --- End Digital Twin Section ---

                # --- XAI EXPLANATION SECTION (Existing Logic) ---
                st.divider() # Separator before XAI
                with st.expander("How was this score calculated? Explain the features.", expanded=False):
                    if xai_artifacts_loaded and shap_explainer: # Check if artifacts & explainer loaded
                        with st.spinner("Generating explanation details..."):
                            # Preprocess data locally for SHAP
                            scaled_data_np, imputed_df_shap = preprocess_data_for_shap(input_data, FEATURES, loaded_scaler, loaded_imputer) # Assuming this function is defined
                            if scaled_data_np is not None and imputed_df_shap is not None:
                                try:
                                    print("Calculating SHAP values...")
                                    shap_values_result = shap_explainer.shap_values(scaled_data_np)
                                    print("SHAP values calculation complete.")

                                    # Determine correct SHAP values instance and expected value (handle different outputs)
                                    if isinstance(shap_values_result, list) and len(shap_values_result) == 2:
                                        shap_values_instance = shap_values_result[1][0] # Class 1 for probability
                                        if isinstance(shap_explainer.expected_value, (list, np.ndarray)) and len(shap_explainer.expected_value) > 1:
                                            expected_value = float(shap_explainer.expected_value[1])
                                        else: expected_value = float(shap_explainer.expected_value)
                                    elif isinstance(shap_values_result, np.ndarray) and shap_values_result.ndim == 2 :
                                        shap_values_instance = shap_values_result[0] # Assuming single output probability
                                        expected_value = float(shap_explainer.expected_value)
                                    elif isinstance(shap_values_result, np.ndarray) and shap_values_result.ndim == 1 and len(shap_values_result) == len(FEATURES):
                                        shap_values_instance = shap_values_result # Direct values
                                        expected_value = float(shap_explainer.expected_value)
                                    else:
                                        raise ValueError(f"Cannot interpret shap_values format: {type(shap_values_result)}")

                                    # === Generate Text Summary of SHAP Explanation ===
                                    plot_summary_text = "Explanation Summary:\n"
                                    try:
                                        feature_values = imputed_df_shap.iloc[0].values
                                        contribs = list(zip(FEATURES, feature_values, shap_values_instance))
                                        contribs.sort(key=lambda x: abs(x[2]), reverse=True) # Sort by impact magnitude
                                        plot_summary_text += f"- Base Score (Average): {expected_value:.4f}\n"
                                        plot_summary_text += f"- Final Predicted Score: {risk_score:.4f}\n"
                                        plot_summary_text += "- Top Feature Contributions:\n"
                                        for i, (feature, val, shap_val) in enumerate(contribs[:5]): # Show top 5
                                            direction = "increases risk" if shap_val > 0 else "decreases risk"
                                            plot_summary_text += f"  - {feature} ({val:.2f}): {'+' if shap_val >=0 else ''}{shap_val:.4f} ({direction})\n"
                                        # Store summary for potential use by chatbot
                                        st.session_state.plot_summary_text = plot_summary_text # Store for later use if needed
                                        print("DEBUG: Generated Plot Summary Text:\n", plot_summary_text)
                                    except Exception as summary_e:
                                        print(f"Error generating plot summary text: {summary_e}")
                                        plot_summary_text = "Error generating summary."
                                    # === End Text Summary Generation ===

                                    # === Generate and Display Waterfall Plot ===
                                    st.markdown("##### Feature Contribution Plot (SHAP Waterfall Plot)")
                                    st.caption("Features pushing score up (red) or down (blue) from base to final.")
                                    try:
                                        shap_explanation = shap.Explanation(values=shap_values_instance,
                                                                            base_values=expected_value,
                                                                            data=imputed_df_shap.iloc[0].values,
                                                                            feature_names=FEATURES)
                                        fig_waterfall, ax_waterfall = plt.subplots()
                                        shap.waterfall_plot(shap_explanation, show=False)
                                        fig_waterfall.tight_layout()
                                        st.pyplot(fig_waterfall, bbox_inches='tight', use_container_width=True)
                                        plt.close(fig_waterfall) # Close plot to free memory
                                        print("DEBUG: Rendered Waterfall plot.")
                                    except Exception as waterfall_e:
                                        st.warning(f"Could not render Waterfall Plot: {waterfall_e}", icon="⚠️")
                                        print(f"Error rendering waterfall plot: {waterfall_e}")
                                    # === End Waterfall Plot ===

                                    # === Chatbot Button - Pass Summary Text ===
                                    st.button("💬 Ask Assistant to Explain This Plot Summary",
                                              key="ask_chatbot_about_plot",
                                              on_click=go_to_chatbot_tool_with_context, # Use helper
                                              args=("plot_explanation", plot_summary_text), # Pass context and summary
                                              use_container_width=True,
                                              disabled=(gemini_model is None))

                                    # === Display Processed Values ===
                                    st.markdown("###### Processed Feature Values Used in Explanation")
                                    st.dataframe(imputed_df_shap.style.format(precision=2))

                                except Exception as shap_e:
                                    st.error(f"Could not generate SHAP explanation: {shap_e}", icon="📉")
                                    print(f"Error during SHAP explanation generation: {shap_e}")
                                    traceback.print_exc()
                            else:
                                st.warning("Could not generate explanation: preprocessing failed.", icon="⚠️")
                    elif not xai_artifacts_loaded:
                        st.warning("Explanation components (model/scaler/imputer) not loaded.", icon="⚠️")
                    else: # Artifacts loaded but explainer failed
                        st.warning("SHAP explainer could not be initialized for the loaded model.", icon="⚠️")
                # --- END: XAI EXPLANATION SECTION ---

            # Handle cases where backend returned an error or invalid response
            elif "error" in result:
                st.error(f"Backend Error: {result.get('error', 'Unknown error')}", icon="☁️")
                print(f"Backend returned error: {result}")
            else:
                st.error("Prediction failed: Invalid response format from backend.", icon="❓")
                st.json(result) # Show the invalid response for debugging

        # Handle specific request exceptions
        except requests.exceptions.Timeout:
            st.error("Connection Error: The request to the backend timed out.", icon="⏱️")
            print("Backend request timed out.")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP Error: {http_err} - Could not get prediction from backend.", icon="☁️")
            print(f"HTTP Error from backend: {http_err}")
            # Try to display detailed error from backend response body
            try:
                error_details = http_err.response.json()
                st.error("Backend Error Details:")
                st.json({"backend_error": error_details})
            except Exception: # If response body isn't JSON or empty
                st.error("Backend Response Content (non-JSON or parse error):")
                st.text(http_err.response.text if http_err.response.text else "(No response body)")
        except requests.exceptions.RequestException as req_err:
            st.error(f"Connection Error: Could not connect to the backend. {req_err}", icon="❌")
            print(f"Network error: {req_err}")
        # Handle any other unexpected errors during the process
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}", icon="💥")
            print(f"Unexpected error in risk_prediction_tool: {e}")
            traceback.print_exc() # Log full traceback for debugging

    st.divider()
    st.caption("Disclaimer: This tool provides estimates based on a prediction model and simulations. It is not medical advice. Always consult with a qualified healthcare professional for any health concerns or decisions.")

def chatbot_tool():
    """Renders the AI Assistant Chatbot interface."""
    if gemini_model is None: st.error("Chatbot unavailable: Check API Key.", icon="🚨"); st.button("← Back to Tools", on_click=go_to_intro); return
    st.title("💬 HealthMate Assistant Chatbot"); st.markdown("Ask about your risk score or general lifestyle suggestions. *Not medical advice.*"); st.divider(); st.button("← Back to Tools", on_click=go_to_intro); st.divider()
    if 'last_risk_score' in st.session_state: risk_score_display = f"{st.session_state.last_risk_score:.4f}" if isinstance(st.session_state.last_risk_score, (int, float)) else 'N/A'; st.info(f"Context: Last risk score **{risk_score_display}**.", icon="💡")
    else: st.info("Ask general questions or run prediction first.", icon="💡")
    if 'chatbot_messages' not in st.session_state: st.session_state.chatbot_messages = [{"role": "assistant", "content": "Hi! How can I help?"}] # Ensure default message if list was cleared
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask about risk, score, lifestyle..."):
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        risk_score_context_val = st.session_state.get('last_risk_score'); input_data_context_val = st.session_state.get('last_prediction_data'); plot_summary_context = st.session_state.get('plot_summary_text_for_chat') # Check if plot summary context needs to be included
        risk_score_context_str = f"{risk_score_context_val:.4f}" if risk_score_context_val is not None else 'not available'
        context_prefix = f"CONTEXT: User score: {risk_score_context_str}. Input: {input_data_context_val if input_data_context_val else 'N/A'}. "
        if plot_summary_context: context_prefix += f"Explanation Summary Provided:\n{plot_summary_context}\n" # Add summary if exists
        system_instructions = "INSTRUCTIONS: Be HealthMate AI. Explain score/summary (use 0.3/0.6 thresholds as examples), give general lifestyle tips related to context/query. DO NOT give medical advice. Always recommend consulting a professional."
        full_prompt = f"{context_prefix}\nUSER QUERY: {prompt}\n\n{system_instructions}"
        with st.chat_message("assistant"):
            message_placeholder = st.empty(); message_placeholder.markdown("Thinking...")
            try:
                if 'gemini_chat_session' not in st.session_state or st.session_state.gemini_chat_session is None: st.session_state.gemini_chat_session = gemini_model.start_chat(history=[]); print("Warning: Gemini chat re-initialized.")
                if st.session_state.gemini_chat_session: response = st.session_state.gemini_chat_session.send_message(full_prompt); assistant_response = response.text; message_placeholder.markdown(assistant_response); st.session_state.chatbot_messages.append({"role": "assistant", "content": assistant_response})
                else: raise ValueError("Gemini chat session unavailable.")
            except Exception as e: st.error(f"Error communicating with AI: {e}", icon="🤖"); error_message = "Sorry, issue responding."; message_placeholder.markdown(error_message); st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message}); print(f"Gemini API Error: {e}"); traceback.print_exc()
        # Clear plot summary context after first use in chat
        if 'plot_summary_text_for_chat' in st.session_state: del st.session_state.plot_summary_text_for_chat


# --- Insulin Site Tracker Tool ---
SITE_GRID = [f'{prefix}{row}{col}' for prefix in ['L', 'R'] for row in range(1, 4) for col in range(1, 4)]
INJECTION_SEQUENCE = SITE_GRID
def get_next_site_in_sequence(last_site_key):
    if not INJECTION_SEQUENCE: return None
    if last_site_key is None or last_site_key not in INJECTION_SEQUENCE: return INJECTION_SEQUENCE[0]
    try: current_index = INJECTION_SEQUENCE.index(last_site_key); next_index = (current_index + 1) % len(INJECTION_SEQUENCE); return INJECTION_SEQUENCE[next_index]
    except (ValueError, IndexError): print(f"Warning: Error finding next site for '{last_site_key}'. Starting over."); return INJECTION_SEQUENCE[0]

# !!! IMPORTANT: CALIBRATE THESE COORDINATES FOR YOUR 'download.jpeg' IMAGE !!!
SITE_DRAW_COORDINATES = { 'L11': (150, 180), 'L12': (190, 180), 'L13': (230, 180), 'L21': (150, 220), 'L22': (190, 220), 'L23': (230, 220), 'L31': (150, 260), 'L32': (190, 260), 'L33': (230, 260), 'R11': (370, 180), 'R12': (410, 180), 'R13': (450, 180), 'R21': (370, 220), 'R22': (410, 220), 'R23': (450, 220), 'R31': (370, 260), 'R32': (410, 260), 'R33': (450, 260)}

def draw_on_abdomen_image(image_path, last_site_key, suggested_site_key, coordinates):
    try:
        if not os.path.exists(image_path): st.error(f"Image file not found: '{image_path}'", icon="🖼️"); return None
        img = Image.open(image_path).convert("RGB"); draw = ImageDraw.Draw(img)
        last_color, suggested_color, text_color = (50, 50, 255, 220), (50, 200, 50, 220), (0, 0, 0)
        radius, outline_width, font_size = 12, 3, 16
        try:
            # Try loading a common sans-serif font first
            font_size = 16
            font = ImageFont.truetype("DejaVuSans.ttf", font_size) # Linux/common font
            print("DEBUG: Loaded DejaVuSans font.")
        except IOError:
            # If first font fails, try the second one
            print("DEBUG: DejaVuSans font not found, trying Arial.")
            try:
                font = ImageFont.truetype("arial.ttf", font_size) # Windows/common font
                print("DEBUG: Loaded Arial font.")
            except IOError:
                # If both specific fonts fail, use Pillow's default font
                print("Warning: Could not load DejaVuSans or Arial font. Using Pillow default.")
                font = ImageDraw.getfont() # Pillow's fallback
        if suggested_site_key in coordinates: x, y = coordinates[suggested_site_key]; draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], outline=suggested_color, width=outline_width); tx, ty = x + radius + 5, y - (font.getbbox('A')[3] // 2); draw.text((tx, ty), f"Next: {suggested_site_key}", fill=suggested_color, font=font)
        if last_site_key in coordinates: x, y = coordinates[last_site_key]; draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=last_color); tx, ty = x + radius + 5, y - (font.getbbox('A')[3] // 2); draw.text((tx, ty), f"Last: {last_site_key}", fill=last_color, font=font)
        return img
    except Exception as e: st.error(f"Error drawing on image: {e}", icon="🎨"); print(f"Error in draw_on_abdomen_image: {e}"); traceback.print_exc(); return None

def post_diagnosis_care_tool():
    st.title("💉 Insulin Site Tracker"); st.markdown("Track injection sites for rotation. Record last site, see suggested next."); st.link_button("Info on Site Rotation", "https://www.diabetes.org.uk/guide-to-diabetes/managing-your-diabetes/treating-your-diabetes/insulin-injections/injecting-sites"); st.divider(); st.button("← Back to Tools", on_click=go_to_intro); st.divider()
    st.subheader("Site Information:"); last_site_display = st.session_state.get('last_injection_site'); suggested_site = get_next_site_in_sequence(last_site_display); st.session_state.suggested_next_site = suggested_site
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Last Recorded Site", last_site_display if last_site_display else "None")
    with col_info2:
        st.metric("Suggested Next Site", suggested_site if suggested_site else "N/A", help="Based on sequence.")
    st.divider(); st.subheader("Record Last Injection Site Used:")
    record_col1, record_col2 = st.columns([3, 1])
    with record_col1: selected_location_record = st.selectbox("Select location used:", SITE_GRID, index=SITE_GRID.index(last_site_display) if last_site_display in SITE_GRID else 0, key="select_location_record", label_visibility="collapsed")
    with record_col2:
        if st.button("Record Site", key="record_site_button", use_container_width=True): st.session_state.last_injection_site = selected_location_record; st.success(f"Recorded {selected_location_record}.", icon="✅"); st.rerun()
    st.divider(); st.subheader("Abdomen Site Map Visualization:"); st.caption("Blue dot = Last | Green Circle = Next Suggestion")
    abdomen_image_path = 'download.jpeg' # Ensure this file exists
    modified_image = draw_on_abdomen_image(abdomen_image_path, st.session_state.get('last_injection_site'), st.session_state.get('suggested_next_site'), SITE_DRAW_COORDINATES)
    if modified_image: st.image(modified_image, caption="Injection site guide", use_container_width=True)
    else: st.warning("Could not display site map.", icon="🖼️")
    st.divider(); st.caption("Note: Follow healthcare provider's advice for injection technique and rotation.")

# --- Main App Logic ---
page_to_render = st.session_state.get('page', 'intro')
if page_to_render == 'intro': intro_page()
elif page_to_render == 'risk_prediction': risk_prediction_tool()
elif page_to_render == 'chatbot_tool': chatbot_tool()
elif page_to_render == 'post_diagnosis_care': post_diagnosis_care_tool()
else: st.warning(f"Invalid page state '{page_to_render}', returning to intro."); intro_page()