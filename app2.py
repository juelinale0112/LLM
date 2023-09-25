import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt  # Added for plotting
from langchain.agents import AgentType, create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI

# Define supported file formats for data upload
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

# Function to load data into a pandas DataFrame
@st.cache_data(ttl=7200)  # Cache data for 2 hours
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.name.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

# Streamlit app configuration
st.set_page_config(page_title="LangChain: Chat with Pandas DataFrame")
st.title("LangChain: Chat with Pandas DataFrame")

# Upload data file
uploaded_file = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are supported",
)

# OpenAI API Key input
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Initialize or clear conversation history
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input for the query
if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check if OpenAI API key is provided
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Load data into a DataFrame
    if uploaded_file:
        df = load_data(uploaded_file)  # Define the DataFrame here

    # Initialize LangChain LLM model
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo-0613",
        openai_api_key=openai_api_key,
        streaming=True,
    )

    # Create a pandas dataframe agent
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,  # Pass the DataFrame here
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    # Generate and display the response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])

        # Check if the response is empty (bot couldn't understand)
        if not response:
            response = "I'm sorry, I couldn't understand your question. Please try again."
        else:
            # Check if the user wants a graph
            if "plot" in response.lower() or "chart" in response.lower():
                # Example: Generate a bar chart
                st.write("Generating a sample bar chart...")
                data = {"Category": ["A", "B", "C"], "Value": [3, 6, 2]}
                sample_df = pd.DataFrame(data)
                plt.bar(sample_df["Category"], sample_df["Value"])
                st.pyplot(plt)
                response += "\nHere's a sample bar chart."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
