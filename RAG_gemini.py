import os
import logging
import streamlit as st
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Example: Ensure API keys are set properly
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
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

from RAG_evaluation evaluate_retrieval, evaluate_generation, evaluate_hallucination

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Streamlit Title
st.title("RAG Application using Gemini Pro")

# Load PDF
pdf_path = "my_paper.pdf"  # Ensure this exists in the working directory
if not os.path.exists(pdf_path):
    st.error(f"File not found: {pdf_path}")
    st.stop()

loader = PyPDFLoader(pdf_path)
data = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Create vector database
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    persist_directory="chroma_db"
)

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
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})

    # Display response
    answer = response.get("answer", "No response generated.")
    st.write("**Answer:**", answer)

    # Run evaluations
    retrieval_results = evaluate_retrieval(query, retriever)
    generation_results = evaluate_generation(query, response)
    hallucination_results = evaluate_hallucination(query, response,retriever.get_relevant_documents(query))

    # Display evaluation results
    st.subheader("Evaluation Metrics")
    st.json(retrieval_results)
    st.json(generation_results)
    st.json(hallucination_results)
    # st.write("**Retrieval Evaluation:**", retrieval_results)
    # st.write("**Generation Evaluation:**", generation_results)
    # st.write("**Hallucination Detection:**", hallucination_results)

    # # Log metrics in the console for debugging
    # print("Retrieval Results:", retrieval_results)
    # print("Generation Results:", generation_results)
    # print("Hallucination Results:", hallucination_results)
