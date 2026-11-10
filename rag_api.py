from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(title="RAG Q&A API")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
model = init_chat_model("google_genai:gemini-3.5-flash-lite", timeout=30)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context. If the answer is not in the "
               "context, say you cannot find it. Keep it to 1-3 sentences."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | model | StrOutputParser()

store = None  # holds the vector store for the current document


class Document(BaseModel):
    text: str


class Question(BaseModel):
    question: str


@app.post("/documents")
def add_document(doc: Document):
    global store
    chunks = splitter.split_text(doc.text)
    store = InMemoryVectorStore.from_texts(chunks, embedding=embeddings)
    return {"chunks_indexed": len(chunks)}


@app.post("/ask")
def ask(q: Question):
    if store is None:
        raise HTTPException(status_code=400, detail="Add a document first")
    docs = store.similarity_search(q.question, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    answer = chain.invoke({"context": context, "question": q.question})
    return {"answer": answer, "sources": [d.page_content for d in docs]}