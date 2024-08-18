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
import tempfile

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
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        if file.name.endswith(".pdf"):
            reader = PdfReader(temp_file_path)
            text = "".join(page.extract_text() for page in reader.pages)
        elif file.name.endswith(".docx"):
            doc = docx.Document(temp_file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file.name.endswith(".txt"):
            with open(temp_file_path, 'r') as f:
                text = f.read()
        else:
            return ""

        os.unlink(temp_file_path)
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
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        logger.info("Vector store created and saved successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

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
        new_db = st.session_state.vector_store
        if new_db is None:
            return "No documents have been processed. Please upload and process documents first."

        docs = new_db.similarity_search(user_question, k=4)  # Increase k for more context
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer this question based on the provided context."
        
        Context:
        {context}
        
        Question: {user_question}
        
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model.invoke(prompt)
        
        return response.content
    except Exception as e:
        logger.error(f"Error in user_input: {str(e)}")
        return "An error occurred while processing your question. Please try again."

def setup_agent():
    search = TavilySearchResults()
    tools = [
        Tool(
            name="Document QA System",
            func=user_input,
            description="Always use this tool first for answering questions about the uploaded documents."
        ),
        Tool(
            name="Search",
            func=search.run,
            description="Use this tool only if the Document QA System couldn't provide an answer."
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

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    st.sidebar.title("Upload Documents")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
    other_docs = st.sidebar.file_uploader("Upload Other Document Files (docx, txt)", accept_multiple_files=True)

    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents..."):
            raw_text = get_text(pdf_docs, other_docs)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            if st.session_state.vector_store:
                st.sidebar.success("Documents processed successfully!")
            else:
                st.sidebar.error("Failed to process documents. Please try again.")

    st.subheader("Ask a Question")
    user_question_input = st.text_input("Your Question", key="user_question_input", placeholder="Enter your question here...")

    if st.button("Process Question", key="process_button"):
        if user_question_input:
            with st.spinner("Processing..."):
                if st.session_state.vector_store is None:
                    st.error("Please upload and process documents first.")
                else:
                    user_response = user_input(user_question_input)  # Direct call to user_input
                    st.session_state.conversation_history.append({"question": user_question_input, "response": user_response})

    st.subheader("Conversation History")
    for entry in reversed(st.session_state.conversation_history):
        st.markdown(f"<div class='stUserInput'>You : {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='stBotResponse'>Bot: {entry['response']}</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Clear Conversation History", type="secondary", help="This will clear the conversation history."):
        st.session_state.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()
