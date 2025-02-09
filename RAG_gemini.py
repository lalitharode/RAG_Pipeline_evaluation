
import os
import logging
import hashlib
import pickle
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging to save logs to a file
log_file_path = "rag_application.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)

# Ensure API keys are set properly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("Missing Google API key! Set it in .env or environment variables.")
    st.error("Missing Google API key! Set it in .env or environment variables.")
    st.stop()

# Import LangChain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from RAG_evaluation import evaluate_retrieval, evaluate_generation, evaluate_hallucination

# Streamlit Title
st.title("RAG Application using Gemini Pro")

# PDF file path
pdf_path = "my_paper.pdf"
if not os.path.exists(pdf_path):
    logging.error(f"File not found: {pdf_path}")
    st.error(f"File not found: {pdf_path}")
    st.stop()

# Function to compute file hash
def compute_file_hash(file_path):
    """Compute SHA256 hash of the file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

# Hash storage file
hash_file = "chroma_db/file_hash.pkl"
vector_db_path = "chroma_db"

# Load previous hash if exists
previous_hash = None
if os.path.exists(hash_file):
    with open(hash_file, "rb") as f:
        previous_hash = pickle.load(f)

# Compute current hash
current_hash = compute_file_hash(pdf_path)

# Check if the vector database already exists and if the document has changed
if os.path.exists(vector_db_path) and previous_hash == current_hash:
    logging.info("Vector database already exists and document has not changed. Loading existing database.")
    vectorstore = Chroma(persist_directory=vector_db_path, embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
else:
    logging.info("Creating vector database...")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create and persist vector database
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=vector_db_path
    )

    # Save the new hash
    os.makedirs("chroma_db", exist_ok=True)
    with open(hash_file, "wb") as f:
        pickle.dump(current_hash, f)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Load Gemini Pro model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

# Define prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Chat input
query = st.chat_input("Ask me anything:")

if query:
    logging.info(f"Received query: {query}")

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    # Display response
    answer = response.get("answer", "No response generated.")
    logging.info(f"Generated answer: {answer}")
    st.write("**Answer:**", answer)

    # Run evaluations
    retrieval_results = evaluate_retrieval(query, retriever)
    generation_results = evaluate_generation(query, response)
    hallucination_results = evaluate_hallucination(query, response, retriever.get_relevant_documents(query))

    logging.info(f"Retrieval Evaluation: {retrieval_results}")
    logging.info(f"Generation Evaluation: {generation_results}")
    logging.info(f"Hallucination Detection: {hallucination_results}")

    # Display evaluation results
    # st.subheader("Evaluation Metrics")
    # st.json(retrieval_results)
    # st.json(generation_results)
    # st.json(hallucination_results)
