# Key changes and explanations:

# I've added imports for the agent and tools:
# pythonCopyfrom langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
# from langchain_community.tools.tavily_search import TavilySearchResults

# I've created a new function setup_agent() that initializes the agent with two tools: 
#Tavily Search and the Document QA System.
# The user_input() function now returns the response text directly, without checking for 
#"Answer is not available in the context".
# I've added a new function process_question() that uses the agent to process the user's question.
# In the main() function, I've replaced the direct call to user_input() with process_question().
# The agent is now stored in the Streamlit session state to avoid reinitializing it on every rerun.

# This modified version uses an agent that can decide whether to use the document QA system or 
#perform a web search using Tavily, based on the question asked. If the document QA system doesn't 
#have enough information, the agent can automatically switch to using the Tavily search tool.

# FAISS is a library for efficient similarity search and clustering of dense vectors. It is often used 
# as a component within vector databases to provide efficient vector search capabilities.


import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pickle
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
# from langchain_community.tools import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure you have set the TAVILY_API_KEY in your .env file
tavily_api_key = os.getenv("TAVILY_API_KEY")

def get_text_from_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        full_text = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(full_text)
    elif file.name.endswith(".txt"):
        return file.getvalue().decode("utf-8")
    else:
        return " NONE "

def get_text(pdf_docs, other_docs):
    text = ""
    for pdf in pdf_docs:
        text += get_text_from_file(pdf)
    for doc in other_docs:
        text += get_text_from_file(doc)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    with open("faiss_index.pickle", "wb") as f:
        pickle.dump(vector_store, f)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "I don't have enough information to answer this question based on the provided context.", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        with open("faiss_index.pickle", "rb") as f:
            new_db = pickle.load(f)
    except FileNotFoundError:
        return "Vector store not available. Please regenerate the document index."

    if not new_db:
        return "Vector store not available. Please regenerate the document index."

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def setup_agent():
    # Set up the base tools
    search = TavilySearchResults()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world"
        ),
        Tool(
            name="Document QA System",
            func=user_input,
            description="useful for when you need to answer questions about the documents that have been uploaded"
        )
    ]

    # Set up the agent
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    return agent

def process_question(agent, question):
    response = agent.run(question)
    return response

def main():
    st.set_page_config("Chat PDF", layout="wide", page_icon=":robot:")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f08080;
            color: #333333;
        }
        .stHeader {
            background-color: #007bff;
            color: #ffffff;
            padding: 10px;
        }
        .stSubheader {
            font-weight: bold;
            margin-top: 20px;
            color: #007bff;
        }
        .stUserInput {
            background-color: #e6e6e6;
            padding: 10px;
            border-radius: 5px;
        }
        .stBotResponse {
            background-color: #eee8aa; 
            padding: 10px;
            border-radius: 5px;
            color: #155724;
        }
        .stClearButton {
            background-color: #dc3545;
            color: #ffffff;
            border-radius: 5px;
            padding: 5px 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'agent' not in st.session_state:
        st.session_state.agent = setup_agent()

    st.sidebar.title("Upload Documents")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    other_docs = st.sidebar.file_uploader("Upload Other Document Files (docx, txt)", accept_multiple_files=True)

    st.subheader("Conversation History")
    if st.session_state.conversation_history:
        chat_log = ""
        for entry in st.session_state.conversation_history:
            chat_log += f"<div class='stUserInput'>You : {entry['question']}</div><div class='stBotResponse'>Bot: {entry['response']}</div><br>"
        st.markdown(chat_log, unsafe_allow_html=True)
    else:
        st.write("No conversation history yet. Start asking questions!")

    st.subheader("Ask a Question")
    user_question_input = st.text_input("Your Question", key="user_question_input", placeholder="Enter your question here...")

    process_button = st.button("Process Question", key="process_button")

    if process_button:
        if user_question_input:
            raw_text = get_text(pdf_docs, other_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            
            user_response = process_question(st.session_state.agent, user_question_input)
            
            st.session_state.conversation_history.append({"question": user_question_input, "response": user_response})
            st.markdown(f'<div class="stBotResponse">Bot : {user_response}</div>', unsafe_allow_html=True)
            st.experimental_rerun()

    if st.button("Clear Conversation History", type="secondary", help="This will clear the conversation history."):
        st.session_state.conversation_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()