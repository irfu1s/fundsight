import google.generativeai as genai
import pandas as pd

class ExplanationAgent:
    def __init__(self, model=None):
        self.gemini_model = model
        # We only need the clean data to look up fund names, not for math
        try:
            self.df_raw = pd.read_csv("data/sip_results_active_clean.csv")
        except:
            self.df_raw = pd.DataFrame(columns=["scheme_name"])

    def handle(self, user_input, state):
        user_text = user_input.lower()
        selected_fund = state.context.get("selected_fund")

        # 1. Try to find fund name in user input
        if not self.df_raw.empty:
            for name in self.df_raw['scheme_name'].unique():
                if str(name).lower() in user_text:
                    selected_fund = name
                    state.context["selected_fund"] = name
                    break
        
        # 2. If we still don't know the fund, ask.
        if not selected_fund:
            return "Please mention which fund you want me to explain (e.g., 'Why HDFC Small Cap?')."

        # 3. Use Gemini to generate the explanation
        if self.gemini_model:
            try:
                prompt = (
                    f"You are a financial expert. The user is asking about the mutual fund: '{selected_fund}'. "
                    "Provide a short, 2-sentence explanation of why this fund is generally considered a good investment "
                    "(focus on consistent returns, reputation, or category performance). "
                    "Do not give financial advice, just an educational summary."
                )
                response = self.gemini_model.generate_content(prompt)
                return f"**Analysis for {selected_fund}**\n\n{response.text.strip()}"
            except Exception as e:
                print(f"Explanation AI Error: {e}")

        # 4. Fallback if AI fails
        return (
            f"**Analysis for {selected_fund}**\n\n"
            "This fund is selected based on its high historical consistency and risk-adjusted returns "
            "compared to its peers in the same category."
        )