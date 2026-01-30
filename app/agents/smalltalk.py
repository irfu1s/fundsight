import google.generativeai as genai

class SmallTalkAgent:
    def __init__(self, model=None):
        self.gemini_model = model  # ✅ Store Gemini Model

    def handle(self, text, state):
        # 1. Try Gemini for a natural, friendly response
        if self.gemini_model:
            try:
                prompt = (
                    "You are 'Portfolia', a friendly, warm, and concise financial AI assistant. "
                    "Respond to the user's social message (greeting, thanks, or emotion). "
                    "Rules:\n"
                    "1. If they say 'Hi' or 'Hello', welcome them warmly and mention you can help with **Investment Recommendations** or **SIP Calculations**.\n"
                    "2. If they seem scared, confused, or new to investing, be very empathetic and reassuring. Tell them investing is easy and you will guide them step-by-step.\n"
                    "3. Keep your response short (under 2 sentences).\n"
                    f"User Message: '{text}'"
                )
                
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            
            except Exception as e:
                print(f"SmallTalk AI Error: {e}")
                # Fall through to standard logic if AI fails

        # 2. Standard Fallback Logic (Your original reliable code)
        t = text.lower()

        if any(w in t for w in ["hi", "hello", "hey"]):
            return "Hi 🙂 I can help you plan SIP investments or calculate returns."

        if any(w in t for w in ["thanks", "thank you"]):
            return "You're welcome 🙂"

        if any(w in t for w in ["scared", "afraid", "new", "confused", "nervous"]):
            return (
                "That's completely normal. Investing feels confusing at first. "
                "We'll go step by step."
            )

        return "Tell me what you'd like to do — invest or calculate returns."