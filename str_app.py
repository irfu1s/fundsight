import sys
import os
import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai

# Fix path to look for 'app' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app.state import AgentState
from app.router import route
from app.agents.advisor import AdvisorAgent
from app.agents.calculator import CalculatorAgent
from app.agents.explanation import ExplanationAgent
from app.agents.smalltalk import SmallTalkAgent
from app.agents.fallback import FallbackAgent

# --- 1. CONFIGURATION (Centered Layout) ---
st.set_page_config(page_title="fundsight", page_icon="📈", layout="centered")

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
    <style>
    /* 1. Global Typography: Crisp, high-contrast text */
    html, body, [class*="css"] {
        -webkit-font-smoothing: antialiased;
    }
    p, li, span, div[data-testid="stMarkdownContainer"] p {
        font-size: 15px !important;
        line-height: 1.6 !important;
        color: #E2E8F0 !important;
    }

    /* 2. Bold, Centered Title */
    h1 {
        font-size: 48px !important;
        font-weight: 800 !important;
        text-align: center;
        color: #FFFFFF;
        text-shadow: 0px 4px 10px rgba(0,0,0,0.4);
        margin-bottom: 30px !important;
    }

    /* 3. Sleek, Distinct Chat Bubbles */
    [data-testid="stChatMessage"] {
        background-color: #161B22 !important; /* Dark grey background */
        border-radius: 15px !important;
        padding: 15px 20px !important;
        margin-bottom: 15px !important;
        border: 1px solid #30363D !important; /* Subtle border */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }

    /* 4. Oval, Floating Chat Input Box */
    [data-testid="stChatInput"] {
        border-radius: 30px !important; /* Makes it a pill shape */
        border: 1px solid #30363D !important;
        background-color: #0D1117 !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Fundsight AI 📈")

# --- 2. CONFIGURE GEMINI ---
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model = genai.GenerativeModel('models/gemini-flash-latest')
    except Exception as e:
        st.error(f"Error loading Gemini: {e}")
        model = None
else:
    model = None

# --- 3. LOAD AGENTS ---
@st.cache_resource
def load_agents():
    return {
        "advisor": AdvisorAgent(model=model),
        "calculator": CalculatorAgent(model=model),
        "explanation": ExplanationAgent(model=model), 
        "smalltalk": SmallTalkAgent(model=model),
        "fallback": FallbackAgent(model=model)
    }

agents = load_agents()

# --- 4. STATE ---
if "backend_state" not in st.session_state:
    st.session_state.backend_state = AgentState()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. DISPLAY LOGIC ---
def display_message(content, unique_key):
    if isinstance(content, dict):
        if "text" in content:
            st.markdown(content['text'])

        if "images" in content and content["images"]:
            for i, img_path in enumerate(content["images"]):
                if os.path.exists(img_path):
                    st.image(img_path, caption="Analysis", width="content")
        
        if 'data' in content and not content['data'].empty:
            df = content['data']
            
            if "Year" in df.columns and "Value" in df.columns:
                st.subheader("📈 Wealth Growth")
                # Updated with Plotly Dark Template
                fig = px.line(df, x="Year", y="Value", title="Projected Growth", markers=True, template="plotly_dark")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)') # Transparent background
                st.plotly_chart(fig, width="stretch", key=f"line_chart_{unique_key}")

            elif "Scheme Name" in df.columns:
                st.subheader("📊 Fund Comparison")
                df_table = df.copy()
                if "Scheme Name" in df_table.columns:
                    df_table["Scheme Name"] = df_table["Scheme Name"].astype(str).str.replace("<br>", " ")
                
                st.dataframe(df_table, hide_index=True)
                
                if "Invested" in df.columns and "Final Value" in df.columns:
                    df_melted = df.melt(id_vars=["Scheme Name"], value_vars=["Invested", "Final Value"], 
                                        var_name="Type", value_name="Amount")
                    
                    # Updated with Plotly Dark Template
                    fig = px.bar(df_melted, x="Scheme Name", y="Amount", color="Type", 
                                 barmode="group", title="Investment vs Return",
                                 color_discrete_map={"Invested": "#FFA726", "Final Value": "#66BB6A"},
                                 template="plotly_dark")
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)') # Transparent background
                    
                    st.plotly_chart(fig, width="stretch", key=f"bar_chart_{unique_key}")
    else:
        st.markdown(content)

# --- 6. RENDER HISTORY ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        display_message(message["content"], unique_key=str(i))

# --- 7. CHAT INPUT ---
if user_input := st.chat_input("Ask about SIPs, funds, or returns..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    current_state = st.session_state.backend_state
    
    # Process logic
    intent = route(user_input, current_state, model=model)
    
    with st.chat_message("assistant"):
        # Classic spinner is back!
        with st.spinner("Analyzing data..."):
            if intent == "clarify":
                word_used = "SIP/Lumpsum"
                if "one" in user_input.lower() or "lump" in user_input.lower():
                    word_used = "One-Time Investment"
                elif "sip" in user_input.lower():
                    word_used = "SIP"
                reply = f"Do you want to **Calculate Returns** for {word_used}, or get a **Recommendation**?"
            else:
                agent = agents.get(intent, agents["fallback"])
                reply = agent.handle(user_input, current_state)
            
        # Display the actual result after the spinner disappears
        new_msg_id = len(st.session_state.messages)
        display_message(reply, unique_key=f"new_{new_msg_id}")
        
    st.session_state.messages.append({"role": "assistant", "content": reply})
