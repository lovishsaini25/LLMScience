import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from tools import BuiltInTools, CustomTools

## Set up the Streamlit app
st.set_page_config(page_title="LLM Science Project", page_icon="🔬")
st.title("Advanced Science Problem Solver Using Gemma 2")

# API Key setup
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


builtin_tools = BuiltInTools()
custom_tools = CustomTools(llm)

# Initialize agents
combined_tools = [builtin_tools.arxiv_tool(), builtin_tools.wikipedia_tool(),
                  builtin_tools.search_engine(), custom_tools.logical_tool(),
                  custom_tools.numerical_math_tool(), custom_tools.equation_math_tool()]

## initialize the agents
assistant_agent=initialize_agent(
    tools=combined_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I can assist you with scientific,"
         "mathematical, and reasoning-based questions."}
    ]
    # When we ask to support or reference to the support the answer, it helps to remove hallucination

st.chat_message(st.session_state.messages[0]["role"]).write(st.session_state.messages[0]["content"])

# User interaction
question = st.text_area("Enter your science question:", "What is the speed of light in a vacuum?")

if st.button("Get my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                # Running all agents with ArXiv and reasoning tools
                response = assistant_agent.invoke({"input": question}, {"callbacks": [st_cb]})
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                st.write('### Response:')
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
