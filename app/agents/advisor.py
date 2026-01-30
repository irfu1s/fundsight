import re
import pandas as pd
import xgboost as xgb
import shap
import json
import os
import difflib
import textwrap
import streamlit as st
from tabulate import tabulate
from src.recommender import get_recommendations
import google.generativeai as genai # ✅ NEW: Import for typing if needed

# --- 🛠️ HELPER: MATCH KEY CREATOR ---
def create_match_key(text):
    if not isinstance(text, str): return ""
    text = str(text).lower()
    text = re.sub(r'\b(direct|growth|regular|plan|fund|scheme|option|dividend|idcw|india)\b', '', text)
    return re.sub(r'[^a-z0-9]', '', text)

# --- 🇮🇳 HELPER: INDIAN CURRENCY FORMATTER ---
def format_indian_number(n):
    try:
        n = int(float(n))
        s = str(n)
        if len(s) <= 3: return s
        last_3 = s[-3:]
        rest = s[:-3]
        # Insert commas every 2 digits for the rest
        rest = re.sub(r'\B(?=(\d{2})+(?!\d))', ",", rest)
        return f"{rest},{last_3}"
    except:
        return str(n)

# 🛑 RENAMED TO '_v28' TO FORCE CACHE RELOAD
@st.cache_resource
def load_advisor_resources_v28():
    print("🧠 Loading Advisor AI Model (Fresh V28)...")
    try:
        model = xgb.Booster()
        model.load_model("models/final_xgb_model.json")

        df_features = pd.read_csv("data/sip_features_model_ready.csv")
        df_raw = pd.read_csv("data/sip_results_active_clean.csv")

        # Standardize columns
        df_raw.columns = [c.strip().lower().replace(' ', '_') for c in df_raw.columns]
        df_features.columns = [c.strip().lower().replace(' ', '_') for c in df_features.columns]

        if "scheme_code" in df_raw.columns:
            df_raw["scheme_code"] = df_raw["scheme_code"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        if "scheme_code" in df_features.columns:
            df_features["scheme_code"] = df_features["scheme_code"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

        with open("models/feature_columns.json", "r") as f:
            feature_names = json.load(f)

        return model, df_features, df_raw, feature_names

    except Exception as e:
        print(f"⚠️ Critical Error loading models: {e}")
        return None, None, None, None


class AdvisorAgent:
    def __init__(self, model=None): # ✅ UPDATED: Accepts Gemini model
        self.done = False
        self.gemini_model = model # ✅ Store the Gemini model
        self.model, self.df_features, self.df_raw, self.feature_names = load_advisor_resources_v28()
        
        self.PLOT_MAP = {
            "daily_return_std": "shap_exports/dependence_daily_return_std.png",
            "months": "shap_exports/dependence_months.png",
            "years": "shap_exports/dependence_years.png",
            "total_invested": "shap_exports/dependence_total_invested.png"
        }
        self.FEATURE_MAP = {
            "sharpe_ratio": {"name": "Risk-Adjusted Return", "pos_msg": "is excellent", "neg_msg": "is low"},
            "daily_return_std": {"name": "Daily Stability", "pos_msg": "is very stable", "neg_msg": "is volatile"},
            "max_drawdown": {"name": "Crash Resistance", "pos_msg": "holds up well", "neg_msg": "drops hard"},
            "category_volatility_pct": {"name": "Category Volatility", "pos_msg": "is safer than avg", "neg_msg": "is riskier than avg"},
            "years": {"name": "Duration", "pos_msg": "fits the goal", "neg_msg": "is short"}
        }

    def normalize(self, text: str) -> str:
        text = text.lower()
        replacements = {"yrs": "year", "yr": "year", "years": "year", "mnths": "month", "mont": "month"}
        for k, v in replacements.items(): text = text.replace(k, v)
        return text

    # --- INPUT EXTRACTION ---
    def extract_amount(self, text):
        text = text.replace(",", "")
        if "lakh" in text or "lac" in text:
            m = re.search(r"(\d+(\.\d+)?)\s*l", text)
            return int(float(m.group(1)) * 100000) if m else None
        if "k" in text:
            m = re.search(r"\b(\d{1,3})k\b", text)
            return int(m.group(1)) * 1000 if m else None
        matches = re.findall(r"\b(\d{3,8})\b", text)
        for val in matches:
            if int(val) >= 100: return int(val)
        return None

    def extract_years(self, text):
        # Priority 1: "5 years", "10 yrs"
        m = re.search(r"(\d+)\s*year", text)
        if m: return int(m.group(1))
        
        # Priority 2: Standalone number
        m = re.search(r"\b(\d{1,2})\b", text)
        return int(m.group(1)) if m else None

    def extract_step_up(self, text):
        if re.search(r"\b(no|none|skip|zero)\b", text): return 0
        if text.strip() == "0": return 0
        m = re.search(r"(\d+)\s*%", text)
        if m: return int(m.group(1))
        m = re.search(r"\b(\d{1,2})\b", text)
        if m: return int(m.group(1))
        return None

    def extract_hold_years(self, text):
        m = re.search(r"(?:hold|holding)\s*(?:for\s*)?(\d+)", text)
        if m: return int(m.group(1))
        m = re.search(r"\b(\d{1,2})\b", text)
        if m: return int(m.group(1))
        return None

    def generate_smart_explanation(self, fund_name, risk_tier, shap_values, feature_names):
        impacts = []
        for name, shap_val in zip(feature_names, shap_values):
            if name in self.FEATURE_MAP:
                impacts.append((name, shap_val))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        images_to_show = ["shap_exports/shap_summary.png"] 
        text_response = f"**{fund_name}** ({risk_tier})"

        if impacts:
            top_feat, top_val = impacts[0]
            config = self.FEATURE_MAP[top_feat]
            msg = config['pos_msg'] if top_val > 0 else config['neg_msg']
            text_response += f"\n\nI recommended this because its **{config['name']}** {msg}."
            if top_feat in self.PLOT_MAP:
                images_to_show.append(self.PLOT_MAP[top_feat])

        return {"text": text_response, "images": images_to_show, "data": pd.DataFrame()}

    def handle(self, user_input: str, state) -> str:
        text = self.normalize(user_input)

        # 1. HANDLE "WHY"
        why_triggers = ["why", "reason", "explain", "logic", "how come"]
        if any(t in text for t in why_triggers) and state.last_recommendations is not None:
            try:
                df_rec = pd.DataFrame(state.last_recommendations)
                
                scheme_code = None
                if "scheme_code" in df_rec.columns:
                    scheme_code = str(df_rec.iloc[0]["scheme_code"]).strip().replace('.0', '')

                if not scheme_code:
                    rec_name = str(df_rec.iloc[0].get("fund_name", ""))
                    target_key = create_match_key(rec_name)
                    match = pd.DataFrame()
                    if "match_key" in self.df_raw.columns:
                        match = self.df_raw[self.df_raw["match_key"] == target_key]
                        if match.empty:
                            all_keys = self.df_raw["match_key"].astype(str).tolist()
                            close = difflib.get_close_matches(target_key, all_keys, n=1, cutoff=0.6)
                            if close: match = self.df_raw[self.df_raw["match_key"] == close[0]]
                    
                    if not match.empty:
                        scheme_code = str(match.iloc[0]["scheme_code"]).strip().replace('.0', '')
                    else:
                        return f"⚠️ Explanation unavailable: Could not link '{rec_name}' to analysis data."

                feat_row = self.df_features[self.df_features["scheme_code"].astype(str) == scheme_code]
                if feat_row.empty: return f"⚠️ Analysis data missing for scheme_code {scheme_code}."

                res_row = self.df_raw[self.df_raw["scheme_code"].astype(str) == scheme_code]
                fund_name = res_row.iloc[0]["scheme_name"] if not res_row.empty else scheme_code

                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(feat_row[self.feature_names])

                return self.generate_smart_explanation(
                    fund_name=fund_name,
                    risk_tier=res_row.iloc[0].get("category_risk_tier", "Unknown") if not res_row.empty else "Unknown",
                    shap_values=shap_values[0],
                    feature_names=self.feature_names
                )
            except Exception as e:
                return f"⚠️ Error generating explanation: {str(e)}"

        # 2. LOGIC FLOW
        reset_triggers = ["reset", "start over", "new plan", "another", "restart"]
        is_switching_mode = any(t in text for t in ["invest", "sip", "lumpsum"]) and state.amount is not None
        
        if any(t in text for t in reset_triggers) or is_switching_mode:
             state.reset_advisor()

        # --- INPUT EXTRACTION (FIXED: 20 Lakh Year Bug) ---
        amount_found_this_turn = False
        if state.amount is None:
            amt = self.extract_amount(text)
            if amt: 
                state.amount = amt
                amount_found_this_turn = True
        
        if state.sip_years is None:
            # If we just extracted "20 Lakh", ignore the number "20" for years
            if amount_found_this_turn:
                 if "year" in text or "yr" in text:
                      yrs = self.extract_years(text)
                      if yrs: state.sip_years = yrs
            else:
                yrs = self.extract_years(text)
                if yrs: state.sip_years = yrs

        # A. Type
        if getattr(state, 'inv_type', None) is None:
            if any(t in text for t in ["lumpsum", "one time", "onetime", "fixed", "bulk"]):
                state.inv_type = "lumpsum"
            elif "sip" in text or "monthly" in text or "month" in text:
                state.inv_type = "sip"
            if state.inv_type is None and state.amount is not None:
                 state.inv_type = "sip"
            if state.inv_type is None:
                 return "Do you want to start a **SIP** (Monthly) or **One-Time** (Lumpsum) investment?"

        # B. Amount
        if state.amount is None:
            if state.inv_type == "lumpsum": return "How much do you want to invest? (e.g. 1 Lakh)"
            else: return "How much do you want to invest monthly?"

        # C. Years
        if state.sip_years is None:
            return "For how many years will you invest?"
        
        # D. Step Up
        if state.inv_type == "sip" and getattr(state, 'step_up', None) is None:
            if "year" in text or text.isdigit():
                 if "no" not in text and "%" not in text and "step" not in text:
                       return "Do you want to increase the SIP yearly? (e.g., type '10%' or 'no')"
            
            val = self.extract_step_up(text)
            if val is not None:
                state.step_up = val
                return f"Applying {val}% yearly step-up. After the SIP ends, for how many years do you want to HOLD? (Enter 0 to skip)"
            return "Do you want to increase the SIP yearly? (e.g., type '10%' or 'no')"

        # E. Hold Years
        if state.inv_type == "sip" and state.hold_years is None:
            if any(w in text for w in ["no", "skip", "zero", "none", "0"]):
                state.hold_years = 0
            else:
                yrs = self.extract_hold_years(text)
                if yrs is not None:
                    state.hold_years = yrs
                else:
                    return "After your SIP ends, for how many years would you like to hold? (Enter 0 to skip)"

        # F. Risk
        if state.risk is None:
            for r in ["low", "medium", "high"]:
                if r in text: 
                    state.risk = r
                    break
            if state.risk is None: return "What is your risk preference? (Low / Medium / High)"

        # G. Category
        if state.category is None:
            if "index" in text or "nifty" in text: state.category = "index_fund"
            elif "large" in text: state.category = "large_cap"
            elif "mid" in text: state.category = "mid_cap"
            elif "small" in text: state.category = "small_cap"
            elif "multi" in text or "flexi" in text: state.category = "multi_cap"
            else: return "Which category? (Large / Mid / Small / Multi / Index)"

        # 4. RECOMMENDATION
        if not self.done:
            if getattr(state, 'step_up', None) is None: state.step_up = 0
            if getattr(state, 'hold_years', None) is None: state.hold_years = 0

            try:
                recs = get_recommendations(
                    category=state.category,
                    risk_level=state.risk,
                    amount=state.amount,
                    duration_years=state.sip_years,
                    investment_type=state.inv_type,
                    hold_years=state.hold_years,
                    step_up_percent=state.step_up
                )

                state.last_recommendations = pd.DataFrame(recs).copy()
                self.done = False 
                
                type_label = "One-Time" if state.inv_type == "lumpsum" else "SIP"
                step_msg = f" (w/ {state.step_up}% step-up)" if (state.inv_type == "sip" and state.step_up > 0) else ""
                disclaimer = ""
                if state.category == "small_cap" and state.risk == "low":
                    disclaimer = "\n\n*(Note: These are 'Low Risk' relative to other Small Caps. Small Caps are generally volatile.)*"

                df_result = pd.DataFrame(recs)
                if not df_result.empty:
                    state.context["selected_fund"] = df_result.iloc[0]["fund_name"]

                display_df = df_result[[
                    "fund_name", "predicted_cagr", "total_invested", "final_value", "projected_return_percent"
                ]].copy()
                
                # Wrapping & Cleanup
                display_df['fund_name'] = display_df['fund_name'].astype(str)
                display_df['fund_name'] = display_df['fund_name'].str.replace(r' - -', '', regex=True)
                display_df['fund_name'] = display_df['fund_name'].str.replace(r' - Direct Plan', '', regex=True)
                display_df['fund_name'] = display_df['fund_name'].str.replace(r' - Growth Option', '', regex=True)
                
                def wrap_name_for_chart(name, width=15, max_lines=2):
                    parts = textwrap.wrap(name, width=width)
                    if len(parts) > max_lines:
                        parts = parts[:max_lines]
                        parts[-1] += "…"
                    return "<br>".join(parts)

                display_df["fund_name"] = display_df["fund_name"].apply(wrap_name_for_chart)

                # ✅ FORMAT NUMBERS (Indian Format 20,00,000)
                for col in ["total_invested", "final_value"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{format_indian_number(x)}")
                
                if "projected_return_percent" in display_df.columns:
                     display_df["projected_return_percent"] = display_df["projected_return_percent"].apply(lambda x: f"{x:.2f}%")

                display_df.columns = ["Scheme Name", "CAGR %", "Invested", "Final Value", "Gain %"]

                response_payload = {
                    "text": (
                        f"✅ **Top {state.category.replace('_', ' ').title()} Funds** ({type_label}{step_msg}):\n\n"
                        f"I have analyzed the market and found these top performers for you based on your {state.risk} risk profile."
                        f"{disclaimer}\n\n"
                        "Type  **'reset'** to start over."
                    ),
                    "data": display_df 
                }

                state.reset_advisor()
                state.active_agent = None  
                
                return response_payload

            except Exception as e:
                state.active_agent = None
                return f"⚠️ Error fetching data: {str(e)}. Please try again."

        return "Would you like to try another plan? Type 'reset' to start over."