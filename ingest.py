import json
import os
import unicodedata
import re
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Cargar variables de entorno
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # opcional según tu config

# Inicializar Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "animales-rag"

# Crear índice si no existe
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Conectar al índice
index = pc.Index(index_name)

# Inicializar embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Función para normalizar IDs
def normalize_id(text):
    text_ascii = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text_ascii = re.sub(r'[^a-zA-Z0-9_-]', '_', text_ascii)
    return text_ascii.lower()

# Leer JSONL y preparar datos
docs = []
metadatas = []
ids = []

with open("animales.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        docs.append(data["descripcion"])
        ids.append(normalize_id(data["id"]))
        metadata = data.copy()
        del metadata["descripcion"]
        metadatas.append(metadata)

# Crear vectorstore y subir datos
vecstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

vecstore.add_texts(
    texts=docs,
    metadatas=metadatas,
    ids=ids
)

print("Datos subidos correctamente a Pinecone desde JSONL")
