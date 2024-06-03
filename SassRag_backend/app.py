from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import os
import tempfile
import logging
import fitz 
import docx
import uuid
import random
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

langfuse_handler = CallbackHandler(
    public_key="pk-lf-daf0deac-c538-4036-a9f5-9d15331ed479",
    secret_key="sk-lf-67dfff03-8e28-4779-8240-949b6bb684d5",
    host="https://cloud.langfuse.com"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

class KnowledgeBase(BaseModel):
    card_name: str

class QuestionRequest(BaseModel):
    question: str

class URLRequest(BaseModel):
    url: str

class CollectionRequest(BaseModel):
    name: str

qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
Q_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
VECTOR_SIZE = 1536
distance_func = models.Distance.COSINE
vectors_config = VectorParams(size=VECTOR_SIZE, distance=distance_func)
open_api_key = os.getenv('OPENAI_API_KEY')
embeddings_openai = OpenAIEmbeddings(openai_api_key=open_api_key)

model = ChatOpenAI(api_key=open_api_key, temperature=0)

template = """
Answer the following question as best you can. You have access to the following context:

{context}

Follow these guidelines when responding:
 Only answer the question and nothing else. Don't add any extra information and don't say any wrong information.

 Greetings and Gratitude:
 - If the user greets you, greet them back.
 - If the user thanks you, respond with "I'm glad I could help."

 Context and Uncertainty:
 - If you are unsure of how to answer the question or feel the question is not worded correctly or is out of the context, respond with "Not provided in the context."

 Examples:

 - If the user greets with "Hello" or "Hi" or any similar greeting, reply with "Hello! How can I assist you today?"
 - If the user says "Thank you," reply with "I'm glad I could help."
 - If the user asks "How are you?" or "How's it going?", reply with "I'm doing well, thank you for asking. If you have any questions, go ahead and ask."
 - If the question is "What are you?" or anything similar to that, always reply with "Hey, I'm Cluster, ready to help you with any questions you have about your provided documents."
 
 Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def create_chain(collection_name):
    vector_store = Qdrant(client=Q_client, collection_name=collection_name, embeddings=embeddings_openai)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

@app.get("/collections")
def get_collections():
    try:
        collections = Q_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        return {"collections": collection_names}
    except Exception as e:
        logger.error(f"Error getting collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collections")
def create_collection(request: CollectionRequest):
    try:
        unique_suffix = str(random.randint(1000000000, 9999999999))
        full_collection_name = f"{request.name}_{unique_suffix}"
        Q_client.create_collection(collection_name=full_collection_name, vectors_config=vectors_config)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{name}")
def delete_collection(name: str):
    try:
        Q_client.delete_collection(collection_name=name)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_link")
def generate_link(kb: KnowledgeBase):
    session_id = str(uuid.uuid4())
    sessions[session_id] = kb.card_name
    full_collection_name = f"{kb.card_name}_{session_id}" 
    return {"link": f"http://demosaasraen.netlify.app/shared/{full_collection_name}"}


@app.get("/shared/{session_id}")
def get_shared_session(session_id: str):
    if session_id in sessions:
        return {"card_name": sessions[session_id]}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.post("/ask/{collection_name}")
def ask_question(collection_name: str, request: QuestionRequest):
    try:
        chain = create_chain(collection_name)
        response = chain.invoke(
            request.question,
            config={"callbacks": [langfuse_handler]}
        )
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file/{collection_name}")
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    try:
        collections = Q_client.get_collections()
        if collection_name not in [i.name for i in collections.collections]:
            Q_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.file.read())
            tmp_file_path = tmp_file.name

        if file.content_type == "text/plain":
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                data = f.read()
        elif file.content_type == "application/pdf":
            data = extract_text_from_pdf(tmp_file_path)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            data = extract_text_from_docx(tmp_file_path)
        else:
            os.remove(tmp_file_path)
            raise HTTPException(status_code=400, detail="Unsupported file type")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(data)

        vector_store = Qdrant(client=Q_client, collection_name=collection_name, embeddings=embeddings_openai)
        vector_store.add_texts(chunks)

        os.remove(tmp_file_path)
        return {"status": "File uploaded and processed successfully."}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

@app.post("/add_url/{collection_name}")
async def add_url(collection_name: str, request: URLRequest):
    try:
        collections = Q_client.get_collections()
        if collection_name not in [i.name for i in collections.collections]:
            Q_client.create_collection(collection_name=collection_name, vectors_config=vectors_config)

        loader = WebBaseLoader(request.url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents_chunks = text_splitter.split_documents(documents)

        vector_store = Qdrant(client=Q_client, collection_name=collection_name, embeddings=embeddings_openai)
        vector_store.add_texts([doc.page_content for doc in documents_chunks])

        return {"status": "URL content added and processed successfully."}
    except Exception as e:
        logger.error(f"Error adding URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# To run the app, use the command: python -m uvicorn app:app --reload
