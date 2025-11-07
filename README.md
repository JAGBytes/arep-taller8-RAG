# Sistema RAG de Información sobre Animales

Este proyecto implementa un sistema de Recuperación Aumentada por Generación (RAG) para consultar información sobre animales. Utiliza Pinecone como base de datos vectorial y modelos de OpenAI para generar respuestas precisas basadas en los datos almacenados.

## 📋 Características

- Búsqueda semántica de información sobre animales
- Respuestas generadas por IA basadas en contexto relevante
- Almacenamiento de vectores para búsquedas eficientes
- Interfaz de línea de comandos interactiva

## 🚀 Requisitos

Antes de comenzar, asegúrate de tener instalado:

- Python 3.8 o superior
- [Pip](https://pip.pypa.io/en/stable/) (gestor de paquetes de Python)
- Una cuenta en [OpenAI](https://platform.openai.com/) para obtener una API key
- Una cuenta en [Pinecone](https://www.pinecone.io/) para el almacenamiento vectorial

## 🔧 Instalación

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tu-usuario/arep-taller8-RAG.git
   cd arep-taller8-RAG
   ```

2. Crea y activa un entorno virtual (recomendado):

   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:
   ```
   OPENAI_API_KEY=tu_api_key_de_openai
   PINECONE_API_KEY=tu_api_key_de_pinecone
   PINECONE_ENVIRONMENT=tu_entorno_pinecone  # opcional
   ```

## 🏃 Ejecución

### 1. Cargar datos a Pinecone

Asegúrate de tener un archivo `animales.jsonl` en la raíz del proyecto con los datos de los animales. Luego ejecuta:

```bash
python ingest.py
```

Este script creará un índice en Pinecone llamado `animales-rag` y cargará los datos.

### 2. Ejecutar el sistema RAG

Para iniciar el sistema de preguntas y respuestas:

```bash
python rag.py
```

Una vez iniciado, podrás hacer preguntas sobre animales y el sistema buscará en la base de conocimiento para darte respuestas precisas.

## 🛠️ Estructura del proyecto

- `rag.py`: Script principal que implementa la lógica del sistema RAG
- `ingest.py`: Script para cargar datos a Pinecone
- `animales.jsonl`: Archivo de datos con información sobre animales
- `requirements.txt`: Dependencias del proyecto
- `.env`: Archivo para variables de entorno (no incluido en el repositorio)

## Evidencia

![alt text](<img/Captura de pantalla 2025-11-06 211258.png>)
![alt text](<img/Captura de pantalla 2025-11-07 135339.png>)

## 📝 Notas adicionales

- Asegúrate de que tu archivo `animales.jsonl` tenga el formato correcto con los campos necesarios.
- El sistema está configurado para usar el modelo `gpt-4o-mini` de OpenAI y `text-embedding-3-small` para los embeddings.
- Puedes ajustar los parámetros de búsqueda en `rag.py` según tus necesidades.
