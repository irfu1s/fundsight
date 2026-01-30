from app.state import AgentState
from app.router import route

from app.agents.advisor import AdvisorAgent
from app.agents.calculator import CalculatorAgent
from app.agents.explanation import ExplanationAgent
from app.agents.smalltalk import SmallTalkAgent
from app.agents.fallback import FallbackAgent


def main():
    state = AgentState()

    advisor = AdvisorAgent()
    calculator = CalculatorAgent()
    explainer = ExplanationAgent()
    smalltalk = SmallTalkAgent()
    fallback = FallbackAgent()

    print("🟢 MutualFundsGPT")
    print("Type 'exit' to quit\n")

    while True:
        user = input("You: ").strip()

        intent = route(user, state)

        if intent == "exit":
            print("Bot: Goodbye!")
            break

        elif intent == "advisor":
            reply = advisor.handle(user, state)

        elif intent == "calculator":
            reply = calculator.handle(user, state)

        elif intent == "explanation":
            reply = explainer.handle(user, state)

        elif intent == "smalltalk":
            reply = smalltalk.handle(user, state)

        else:
            reply = fallback.handle(user, state)

        print("Bot:", reply)


if __name__ == "__main__":
    main()
