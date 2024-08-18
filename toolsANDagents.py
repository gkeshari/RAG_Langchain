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
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
tavily_api_key = os.getenv("TAVILY_API_KEY")

def get_text_from_file(file):
    try:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file.name.endswith(".txt"):
            text = file.getvalue().decode("utf-8")
        else:
            return ""
        return text
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        return ""

def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if os.path.exists("faiss_index"):
            vector_store = FAISS.load_local("faiss_index", embeddings)
            logger.info("Vector store loaded successfully")
            return vector_store
        else:
            logger.warning("Vector store not found")
            return None
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None

def get_text(pdf_docs, other_docs):
    text = ""
    for file in pdf_docs + other_docs:
        text += get_text_from_file(file)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    if not text_chunks:
        logger.warning("No text chunks to process")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logger.info("Vector store created and saved successfully")
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")

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
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    try:
        new_db = load_vector_store()
        if not new_db:
            return "Vector store not available. Please regenerate the document index."

        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        return "An error occurred while processing your question. Please try again."

def setup_agent():
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
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def process_question(agent, question):
    try:
        return agent.run(question)
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return "An error occurred while processing your question. Please try again."

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

    if process_button and user_question_input:
        with st.spinner("Processing..."):
            if pdf_docs or other_docs:
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
