# rag.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "animales-rag"

# Validar variables de entorno
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    print("❌ Error: Faltan variables de entorno en .env")
    exit(1)

# Inicializar
print("🔧 Inicializando sistema RAG...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"❌ El índice '{INDEX_NAME}' no existe")
    exit(1)

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    request_timeout=30,
    max_retries=2
)

print("✅ Sistema listo\n")


def preguntar(texto):
    """Realiza una consulta al sistema RAG"""
    try:
        # Obtener documentos relevantes
        docs = retriever.invoke(texto)
        
        if not docs or all(not doc.page_content for doc in docs):
            print("\n⚠️ No se encontró información relevante.\n")
            return
        
        # Construir contexto detallado con toda la metadata
        contexto_partes = []
        for doc in docs:
            meta = doc.metadata
            info = f"""Animal: {meta.get('nombre', 'Desconocido')}
Nombre científico: {meta.get('nombre_cientifico', 'N/A')}
Clasificación: {meta.get('clasificacion', 'N/A')}
Descripción: {doc.page_content}
Hábitat: {meta.get('habitat', 'N/A')}
Dieta: {meta.get('dieta', 'N/A')}
Tamaño promedio: {meta.get('tamano_promedio', 'N/A')}
Peso promedio: {meta.get('peso_promedio', 'N/A')}
Esperanza de vida: {meta.get('esperanza_vida', 'N/A')}
Estado de conservación: {meta.get('estado_conservacion', 'N/A')}"""
            contexto_partes.append(info)
        
        contexto = "\n\n---\n\n".join(contexto_partes)
        
        # Crear prompt optimizado
        prompt = f"""Eres un asistente experto en animales. Responde la pregunta del usuario usando SOLO la información del contexto proporcionado.

REGLAS:
- Si la información está en el contexto, responde de forma clara, natural y conversacional
- Si el contexto tiene información parcialmente relacionada, menciónala
- Si el contexto NO tiene información sobre lo que pregunta, di claramente "No tengo información sobre [tema] en mi base de datos"
- Usa los nombres comunes de los animales en tu respuesta
- Sé amigable y útil

CONTEXTO:
{contexto}

PREGUNTA: {texto}

RESPUESTA:"""

        # Generar respuesta
        respuesta = llm.invoke([
            {"role": "system", "content": "Eres un asistente experto en animales."},
            {"role": "user", "content": prompt}
        ])
        
        # Mostrar respuesta
        print(f"\n💬 {respuesta.content}\n")
        
        # Mostrar fuentes
        fuentes = ", ".join([
            doc.metadata.get('nombre', doc.metadata.get('id', '?').replace('animal_', ''))
            for doc in docs
        ])
        print(f"📚 Consultados: {fuentes}\n")
            
    except Exception as e:
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    print("="*60)
    print("🤖 Sistema RAG - Base de Conocimiento de Animales")
    print("="*60)
    print("Escribe 'salir' para terminar\n")
    
    while True:
        try:
            pregunta = input("💬 Pregunta: ").strip()
            
            if not pregunta:
                continue
            
            if pregunta.lower() in ["salir", "exit", "quit"]:
                print("\n👋 ¡Hasta luego!\n")
                break
            
            preguntar(pregunta)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!\n")
            break
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}\n")