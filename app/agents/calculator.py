import re
import pandas as pd
import streamlit as st
import google.generativeai as genai # ✅ NEW: Import for typing/future use
from src.recommender import simulate_sip_logic, simulate_lumpsum_logic, normalize_scheme_name

# ================= 🇮🇳 HELPER: INDIAN CURRENCY FORMATTER ================= #
def format_indian_number(n):
    try:
        n = float(n)  # Ensure it's a number
        n = int(round(n))  # Round to nearest integer
        s = str(n)
        if len(s) <= 3: return s
        last_3 = s[-3:]
        rest = s[:-3]
        # Insert commas every 2 digits for the rest
        rest = re.sub(r'\B(?=(\d{2})+(?!\d))', ",", rest)
        return f"{rest},{last_3}"
    except:
        return str(n)

# ================= 🚀 1. CACHED DATA LOADER ================= #
# Using 'v3' as per your optimized version
@st.cache_resource
def load_calculator_resources_v3():
    print("🧠 Loading Calculator Data (Optimized)...")
    PRED_PATH = "data/sip_predictions.csv"
    try:
        df = pd.read_csv(PRED_PATH)
        
        # Normalize & Create Lookup
        df["base_name"] = df["scheme_name"].apply(normalize_scheme_name)
        df = df.sort_values("predicted_cagr_percent", ascending=False)
        unique_funds = df.drop_duplicates(subset=["base_name"]).copy()
        
        fund_lookup = unique_funds.set_index("base_name")[["scheme_name", "predicted_cagr_percent"]].to_dict(orient="index")
        
        # Tokenization for Search
        stopwords = {"fund", "plan", "growth", "option", "regular", "scheme", "bonus", "idcw", "dividend", "india"}
        
        def tokenize_func(text):
            clean = re.sub(r"[^a-z0-9\s]", "", text.lower())
            # ✅ OPTIMIZATION: Return a SET immediately
            return {w for w in clean.split() if w not in stopwords}
            
        # ✅ OPTIMIZATION: Store tokens as Sets in memory once. 
        fund_index = [{"name": base, "tokens": tokenize_func(base)} for base in fund_lookup.keys()]
        
        return fund_lookup, fund_index, tokenize_func

    except Exception as e:
        print(f"⚠️ Error loading Calculator CSV: {e}")
        return {}, [], None

# ================= CALCULATOR AGENT ================= #

class CalculatorAgent:
    def __init__(self, model=None): # ✅ UPDATED: Accepts Gemini model
        self.current_options = []
        self.model = model # ✅ Store the Gemini model (Logic remains unchanged)
        # Load from Cache
        self.fund_lookup, self.fund_index, self.tokenize = load_calculator_resources_v3()

    # --- HELPERS ---
    def _extract_amount(self, text):
        text = text.replace(",", "")
        if "lakh" in text or "lac" in text:
            m = re.search(r"(\d+(\.\d+)?)\s*l", text)
            return int(float(m.group(1)) * 100000) if m else None
        if "k" in text:
            m = re.search(r"\b(\d{1,3})k\b", text)
            return int(m.group(1)) * 1000 if m else None
        
        matches = re.findall(r"\b(\d{3,8})\b", text)
        for val in matches:
            if int(val) >= 500: return int(val)
        return None

    def _extract_years(self, text):
        m = re.search(r"(?<!hold\s)(?<!holding\s)\b(\d{1,2})\b", text)
        return int(m.group(1)) if m else None

    def _extract_step_up(self, text):
        if re.search(r"\b(no|none|skip|zero)\b", text): return 0
        if text.strip() == "0": return 0
        m = re.search(r"(\d+)\s*%", text)
        if m: return int(m.group(1))
        m = re.search(r"\b(\d{1,2})\b", text) 
        return int(m.group(1)) if m else None

    def _extract_hold_years(self, text):
        m = re.search(r"(?:hold|holding)\s*(?:for\s*)?(\d+)", text)
        if m: return int(m.group(1))
        return None

    def _get_filtered_matches(self, text):
        if not self.tokenize: return []
        
        user_tokens = self.tokenize(text) # This is now a SET
        if not user_tokens: return []
        
        candidates = []
        
        # ✅ OPTIMIZED LOOP: No set() conversions inside the loop
        for item in self.fund_index:
            fund_tokens = item["tokens"] # Already a SET
            
            # Fast Set Intersection
            overlap = user_tokens & fund_tokens
            
            if not overlap: continue
            
            score = (len(overlap) / len(user_tokens)) + (len(overlap) / len(fund_tokens))
            if any(t in user_tokens for t in ["small", "mid", "large", "index", "nifty", "hdfc", "nippon", "sbi", "axis"]):
                score += 0.5
            
            if score >= 0.8:
                candidates.append((item["name"], score))
        
        return [c[0] for c in sorted(candidates, key=lambda x: x[1], reverse=True)[:5]]

    # --- MAIN HANDLER ---
    def handle(self, user_input: str, state) -> str:
        text = user_input.strip().lower()

        # ---------------------------------------------------------
        # 🆕 1. SMART RESET & INTENT DETECTION
        # ---------------------------------------------------------
        reset_triggers = ["reset", "start over", "new calculation", "restart", "calc"]
        
        is_new_intent = "calculate" in text and len(text.split()) > 1
        
        # 🔒 FIX: If we reset, IMMEDIATELY re-lock the agent to "calculator"
        if any(t in text for t in reset_triggers) or is_new_intent:
            state.reset_calculator()
            state.active_agent = "calculator" 
            self.current_options = []
            # Don't return yet! Fall through to process the input

        # ---------------------------------------------------------

        # 1. FUND SELECTION
        if state.calc_fund is None:
            # A. Numbered Selection
            if self.current_options and text.isdigit():
                idx = int(text) - 1
                if 0 <= idx < len(self.current_options):
                    state.calc_fund = self.current_options[idx]
                    self.current_options = [] 
                    return f"Selected: **{state.calc_fund}**\n\nDo you want to calculate for **SIP** or **One-Time**?"
                return f"Please pick a number between 1 and {len(self.current_options)}."
            
            # B. Confirmation
            if state.pending_fund:
                if any(w in text for w in ["yes", "yup", "yeah"]):
                    state.calc_fund = state.pending_fund
                    state.pending_fund = None
                    return f"Selected: **{state.calc_fund}**\n\nDo you want to calculate for **SIP** or **One-Time**?"
                elif any(w in text for w in ["no", "nope", "nah"]):
                    state.pending_fund = None
                    self.current_options = []
                    return "Okay, please mention the fund name again."

            # C. Search Logic
            matches = self._get_filtered_matches(text)
            
            if not matches:
                # 🛑 FIX for "How much will I have..."
                extracted_amt = self._extract_amount(text)
                if extracted_amt:
                    state.calc_amount = extracted_amt
                    return f"I noted ₹{format_indian_number(extracted_amt)}. Now, please tell me **which Mutual Fund** you want to calculate for?"
                
                return "Please mention the **Mutual Fund name** (e.g., Nippon Small Cap)."
            
            if len(matches) > 1:
                self.current_options = matches
                opts = "\n".join([f"{i+1}. {m}" for i, m in enumerate(matches)])
                return f"I found multiple funds. Type the number to select:\n\n{opts}"
            
            state.pending_fund = matches[0]
            return f"Did you mean '**{matches[0]}**'? (yes / no)"

        # 2. INVESTMENT TYPE
        if state.calc_inv_type is None:
            lumpsum_triggers = ["lumpsum", "one time", "onetime", "one-time", "single", "once", "fixed", "fd"]
            if any(trigger in text for trigger in lumpsum_triggers):
                state.calc_inv_type = "lumpsum"
                if state.calc_amount: return "Got it, One-Time. For how many years will you stay invested?"
                return "Got it, **One-Time**. How much is the total investment?"
            
            if "sip" in text or "monthly" in text or "month" in text:
                state.calc_inv_type = "sip"
                if state.calc_amount: return "Got it, SIP. For how many years?"
                return "Got it, **SIP**. How much per month?"
            
            # Fallback extraction
            amt = self._extract_amount(text)
            if amt:
                state.calc_amount = amt
                state.calc_inv_type = "sip" # Default to SIP
                return f"Assuming SIP of ₹{format_indian_number(amt)}. For how many years?"

            return "Do you want to calculate for **SIP** or **One-Time** investment?"

        # 3. AMOUNT
        if state.calc_amount is None:
            amt = self._extract_amount(text)
            if amt:
                state.calc_amount = amt
                if state.calc_inv_type == "lumpsum":
                    return f"Got it, ₹{format_indian_number(amt)} One-Time. For how many years will you stay invested?"
                return f"Got it, ₹{format_indian_number(amt)}/month. For how many years will you invest?"
            return "Please enter a valid amount (e.g., 5000)."

        # 4. DURATION
        if state.calc_sip_years is None:
            yrs = self._extract_years(text)
            if yrs:
                state.calc_sip_years = yrs
                if state.calc_inv_type == "lumpsum":
                    state.calc_step_up = 0
                    state.calc_hold_years = 0
                else:
                    return "Do you want to step up (increase) the SIP yearly? (Type '%' e.g. '10%' or 'no')"
            else:
                return "For how many years will you invest?"

        # 5. STEP UP (SIP Only)
        if state.calc_inv_type == "sip" and getattr(state, 'calc_step_up', None) is None:
            val = self._extract_step_up(text)
            if val is not None:
                state.calc_step_up = val
                msg = f"Adding {val}% yearly step-up." if val > 0 else "No step-up."
                return f"{msg} Finally, how many years to HOLD after SIP ends? (Enter 0 to skip)"
            return "Do you want to increase the SIP amount yearly? (e.g., '10%' or 'no')"

        # 6. HOLD YEARS (SIP Only)
        if state.calc_inv_type == "sip" and state.calc_hold_years is None:
            if any(w in text for w in ["skip", "no", "none", "zero", "0"]):
                state.calc_hold_years = 0
            else:
                hold = self._extract_hold_years(text)
                if hold is not None:
                    state.calc_hold_years = hold
                else:
                    return "Enter hold years or '0' to skip."

        # 7. CALCULATE
        fund_data = self.fund_lookup.get(state.calc_fund)
        if not fund_data:
             return "⚠️ Error: Could not retrieve data. Try searching again."
             
        cagr = fund_data["predicted_cagr_percent"]
        
        if state.calc_inv_type == "lumpsum":
            invested, final_value, curve = simulate_lumpsum_logic(
                annual_cagr=cagr,
                total_amount=state.calc_amount,
                duration_years=state.calc_sip_years
            )
            type_lbl = "One-Time"
            step_msg = ""
            duration_msg = f"{state.calc_sip_years} years"
        else:
            invested, final_value, curve = simulate_sip_logic(
                annual_cagr=cagr,
                monthly_amount=state.calc_amount,
                sip_years=state.calc_sip_years,
                hold_years=state.calc_hold_years,
                step_up_pct=state.calc_step_up
            )
            type_lbl = "SIP"
            step_msg = f" (w/ {state.calc_step_up}% step-up)" if state.calc_step_up > 0 else ""
            duration_msg = f"{state.calc_sip_years} yrs + {state.calc_hold_years} yrs hold"

        profit = final_value - invested
        return_percent = (profit / invested * 100) if invested > 0 else 0

        # ✅ NEW: Table Format + Indian Currency
        res_text = (
            f"🧾 **Result for {state.calc_fund}**\n"
            f"*Historical CAGR: {cagr:.2f}%*\n\n"

            f"**Investment Summary**\n"
            f"- Mode: **{type_lbl}**\n"
            f"- Investment: **₹{state.calc_amount:,.0f}{step_msg}**\n"
            f"- Duration: **{duration_msg}**\n"
            f"- Total Invested: **₹{invested:,.0f}**\n\n"

            f"**Return Outcome**\n"
            f"- Final Value: **₹{final_value:,.0f}**\n"
            f"- Profit: **₹{profit:,.0f} ({return_percent:.0f}%)**"
        )

        
        # Reset State
        state.reset_calculator()
        self.current_options = []
        state.active_agent = None
        
        # Prepare Data for Graph
        if isinstance(curve, list) and len(curve) > 0 and isinstance(curve[0], dict):
            df_curve = pd.DataFrame(curve)
        else:
            df_curve = pd.DataFrame(curve, columns=["Year", "Invested", "Value"])

        return {
            "text": res_text,
            "data": df_curve 
        }