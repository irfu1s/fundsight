import google.generativeai as genai

class FallbackAgent:
    def __init__(self, model=None):
        self.gemini_model = model  # ✅ Store Gemini Model

    def handle(self, user_input, state):
        # 1. If Gemini is available, try to generate a helpful response
        if self.gemini_model:
            try:
                # System prompt to keep the AI focused on the app's purpose
                system_instruction = (
                    "You are 'Portfolia AI', a helpful financial assistant for a Mutual Fund app. "
                    "Your goal is to help users with investing, SIPs, and calculating returns. "
                    "1. If the user asks a general financial question (e.g., 'What is inflation?', 'Is gold better than equity?'), answer it briefly and accurately. "
                    "2. If the user asks something unrelated (e.g., 'Who is the president?', 'Write a poem'), politely decline and steer them back to mutual funds. "
                    "3. If the user's request is vague, suggest they try: 'I want to invest 5000/month' or 'Calculate returns for SBI Small Cap'. "
                    "Keep answers short (max 3 sentences) and friendly."
                )
                
                full_prompt = f"{system_instruction}\n\nUser Query: {user_input}"
                
                response = self.gemini_model.generate_content(full_prompt)
                return response.text.strip()

            except Exception as e:
                print(f"Fallback AI Error: {e}")
                # If API fails, drop down to the static message below
        
        # 2. Static Fallback (Safety Net)
        return (
            "I'm not sure I understood that. 🤔\n\n"
            "I can help you **Plan an Investment** or **Calculate Returns**.\n"
            "Try saying:\n"
            "- *'Invest ₹5,000 in a SIP'*\n"
            "- *'Calculate returns for Quant Small Cap'*"
        )