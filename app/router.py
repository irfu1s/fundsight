import re
import google.generativeai as genai

def has_exact_word(text: str, word: str) -> bool:
    """
    Returns True ONLY if the word exists as a standalone word.
    Example: 
      - text="high risk", word="hi" -> False (Correct)
      - text="hi there", word="hi" -> True (Correct)
    """
    # \b ensures a word boundary (start/end of string, space, punctuation)
    return re.search(rf"\b{re.escape(word)}\b", text.lower()) is not None

def route(user_input, state, model=None): # ✅ UPDATED: Accepts Gemini model
    text = user_input.lower().strip()

    # =========================================================
    # 0️⃣ PREPARE LISTS (Definitions)
    # =========================================================
    
    # Words that imply Advisor action (Verbs)
    advisor_verbs = [
        "invest", "investment", "investing", "plan", "plans", "planning", 
        "suggest", "suggestion", "recommend", "recommendation", "best", "top", 
        "fund", "funds", "mutual fund", "review", "portfolio"
    ]
    
    # Words that imply Calculator action (Verbs)
    calc_verbs = [
        "calculate", "calc", "calculation", "calculating",
        "return", "returns", "profit", "profits", 
        "value", "valuation", "final value", "growth", "worth"
    ]
    
    # Words that are Neutral/Ambiguous (Nouns)
    # These depend on context!
    ambiguous_nouns = ["sip", "sips", "lumpsum", "one time", "onetime", "fixed", "fd"]

    # Check presence using your helper
    has_adv_verb = any(has_exact_word(text, w) for w in advisor_verbs)
    has_calc_verb = any(has_exact_word(text, w) for w in calc_verbs)
    has_ambiguous_noun = any(has_exact_word(text, w) for w in ambiguous_nouns)

    # =========================================================
    # 1️⃣ EXIT & MASTER RESET
    # =========================================================
    exit_triggers = ["exit", "quit", "bye", "reset", "restart", "start over", "new plan"]
    if any(has_exact_word(text, w) for w in exit_triggers):
        state.reset_all()
        if any(has_exact_word(text, w) for w in ["reset", "restart", "start over", "new plan"]):
            return "smalltalk"
        return "exit"

    # =========================================================
    # 2️⃣ COMBINATION CHECKS (Highest Priority)
    # =========================================================
    # "Invest One Time" -> Advisor
    if has_adv_verb and has_ambiguous_noun:
        state.reset_advisor()
        state.active_agent = "advisor"
        return "advisor"

    # "Calculate One Time" -> Calculator
    if has_calc_verb and has_ambiguous_noun:
        state.reset_calculator()
        state.active_agent = "calculator"
        return "calculator"
    
    # "How much in SIP" -> Calculator
    if "how much" in text and has_ambiguous_noun:
        state.reset_calculator()
        state.active_agent = "calculator"
        return "calculator"

    # =========================================================
    # 3️⃣ CALCULATOR (Strong Triggers)
    # =========================================================
    if "how much" in text:
        state.reset_calculator()
        state.active_agent = "calculator"
        return "calculator"

    if has_calc_verb: 
        state.reset_calculator()
        state.active_agent = "calculator"
        return "calculator"

    # =========================================================
    # 4️⃣ ADVISOR (Strong Triggers)
    # =========================================================
    # We use 'advisor_verbs' here which EXCLUDES "sip/one time"
    # This prevents "One Time" from automatically resetting the Advisor.
    if has_adv_verb:
        state.reset_advisor()
        state.active_agent = "advisor"
        return "advisor"

    # =========================================================
    # 5️⃣ ADVISOR - EXPLANATION (No Reset)
    # =========================================================
    advisor_explain_keywords = [
        "why", "explain", "reason", "logic", "basis", 
        "review", "analysis", "details", "risk", "safe", "safety",
        "how come"
    ]
    if any(has_exact_word(text, w) for w in advisor_explain_keywords):
        state.active_agent = "advisor"
        return "advisor"

    # =========================================================
    # 6️⃣ AMBIGUOUS / CONTEXT HANDLER (The Fix)
    # =========================================================
    if has_ambiguous_noun:
        # A. If we are already talking to someone, STAY with them.
        if state.active_agent is not None:
            return state.active_agent
        
        # B. If IDLE, we don't know what to do -> Ask user.
        return "clarify"

    # =========================================================
    # 7️⃣ SMALL TALK & LOCK-IN
    # =========================================================
    smalltalk_keywords = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "cool", "good", "morning","sure", "evening"]
    if any(has_exact_word(text, w) for w in smalltalk_keywords):
        return "smalltalk"

    # Lock-in for non-keywords (e.g. "Nippon", "5000")
    if state.active_agent is not None:
        return state.active_agent

    # =========================================================
    # 8️⃣ GEMINI INTELLIGENT FALLBACK (✅ NEW)
    # =========================================================
    # If regex failed, and we have a model, let AI decide.
    if model:
        try:
            prompt = (
                f"Classify user intent: '{user_input}'. "
                "Categories: [advisor, calculator, smalltalk, clarify]. "
                "advisor = seeking recommendations, funds, analysis. "
                "calculator = calculating returns, profit, value. "
                "clarify = ambiguous one-word queries like 'sip'. "
                "Return ONLY the category name (lowercase)."
            )
            response = model.generate_content(prompt)
            intent = response.text.strip().lower()
            
            valid_intents = ["advisor", "calculator", "smalltalk", "clarify"]
            if intent in valid_intents:
                # Update state based on AI decision
                if intent == "advisor": 
                    state.active_agent = "advisor"
                elif intent == "calculator": 
                    state.active_agent = "calculator"
                return intent
        except Exception:
            pass # Fail silently back to fallback

    return "fallback"