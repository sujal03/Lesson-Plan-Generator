import os
import json
import uuid
import re
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient
import certifi
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from bson.objectid import ObjectId

# =============================================================================
# Environment Setup and Global Clients
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()

# Fetch the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("OPENAI_API_KEY not set in environment variables.")

# =============================================================================
# MongoDB Functions
# =============================================================================

def get_mongodb_connection() -> MongoClient:
    """Establish and return a connection to MongoDB."""
    try:
        mongo_uri = (
            "mongodb+srv://tamrakarsujal18:qxGOZFKr6jhQN832@cluster0.6bz5j.mongodb.net/"
            "?retryWrites=true&w=majority&appName=Cluster0"
        )
        mongo_client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
        mongo_client.admin.command('ping')  # Test connection
        print("Successfully connected to MongoDB")
        return mongo_client
    except Exception as e:
        raise Exception(f"MongoDB connection failed: {str(e)}")

def push_to_mongo(data: dict) -> str:
    """Push the given dictionary data into MongoDB and return the inserted ID."""
    mongo_client = None
    try:
        mongo_client = get_mongodb_connection()
        db = mongo_client.get_database("LessonPlan")
        collection = db.get_collection("TestingDB")
        result = collection.insert_one(data)
        print(f"Document inserted with id: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        raise Exception(f"Error pushing data to MongoDB: {str(e)}")
    finally:
        if mongo_client:
            mongo_client.close()

# Function to update lesson plan in MongoDB
def update_lesson_plan_in_mongo(document_id: str, lesson_plan: str) -> None:
    """Update the lesson plan in the MongoDB document with the given ID."""
    mongo_client = None
    try:
        mongo_client = get_mongodb_connection()
        db = mongo_client.get_database("LessonPlan")
        collection = db.get_collection("TestingDB")
        result = collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"lesson_plan": lesson_plan}}
        )
        if result.modified_count > 0:
            print(f"Lesson plan updated for document ID: {document_id}")
        else:
            print(f"No document found or no changes made for ID: {document_id}")
    except Exception as e:
        raise Exception(f"Error updating lesson plan in MongoDB: {str(e)}")
    finally:
        if mongo_client:
            mongo_client.close()

# =============================================================================
# PDF Processing Functions
# =============================================================================

def extract_pdf_data(uploaded_file) -> tuple:
    """
    Extract text from an uploaded PDF file using PyPDFLoader.
    
    Returns:
        full_text (str): The combined text from all pages.
        documents (list): The list of Document objects returned by PyPDFLoader.
    """
    try:
        # Read the file content
        input_file = uploaded_file.read()
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(input_file)
            temp_file_path = temp_file.name
        
        # Use PyPDFLoader to load the PDF content
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Combine the page content from each document to form the full text
        full_text = "\n".join(doc.page_content for doc in documents)
        return full_text, documents
    except Exception as e:
        raise Exception(f"Error extracting PDF data: {str(e)}")
    finally:
        # Ensure the temporary file is deleted
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# =============================================================================
# Text Analysis and JSON Handling Functions
# =============================================================================

def analyze_text(text: str, class_data: str) -> str:
    """
    Analyze curriculum text using OpenAI and extract information for the given class/grade.
    
    Returns:
        A JSON string containing the extracted curriculum information.
    """
    prompt = f"""
You are an expert curriculum analyzer. Extract detailed information from the given curriculum document for class/grade "{class_data}" and return it in the following JSON structure only:

{{
    "title": "Unit title",
    "duration": "Duration or 'Not specified'",
    "learningObjectives": [
        "List of learning objectives"
    ],
    "keyConcepts": [
        "List of key topics and concepts"
    ],
    "standards": [
        {{
            "code": "Standard code",
            "description": "Description of the standard"
        }}
    ],
    "assessments": [
        {{
            "type": "Type of assessment (e.g., Quiz, Project)",
            "criteria": "Assessment criteria"
        }}
    ],
    "materials": [
        {{
            "externalLinks": [
                "Array of external resource URLs"
            ],
            "description": "Description of resources"
        }}
    ],
    "tools": [
        "List of tools required"
    ]
}}

Instructions:
1. Extract ONLY information relevant to {class_data}.
2. Return data in the exact JSON structure shown above.
3. Include all available information from the curriculum.
4. Use "Not specified" for any missing information.
5. Return ONLY the JSON object without any additional text or formatting.

Document text:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced educator who creates detailed, practical lesson plans. Return only valid JSON without any markdown formatting."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        result = response.choices[0].message.content.strip()
        # Remove potential markdown code block syntax
        result = result.replace('```json', '').replace('```', '').strip()
        # Validate JSON
        json.loads(result)
        return result
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response from API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error analyzing text: {str(e)}")

# =============================================================================
# Vector Store and Embedding Functions
# =============================================================================

def clean_filename(filename: str) -> str:
    """Clean the filename to remove invalid characters for MongoDB collection names."""
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    cleaned = re.sub(r'_{2,}', '_', cleaned).strip('_')
    cleaned = cleaned[:63] if len(cleaned) > 63 else cleaned
    cleaned = re.sub(r'^[^a-zA-Z0-9]', '', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9]$', '', cleaned)
    return cleaned

def split_document(documents: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)

def get_embedding_function(api_key: str):
    """Return the embedding function using the provided OpenAI API key."""
    return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

def create_vectorstore(chunks: list, embedding_function, file_name: str, vector_store_path: str = "db") -> Chroma:
    """
    Create a Chroma vector store from document chunks.
    
    Removes duplicate documents based on unique IDs.
    """
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    unique_ids = set()
    unique_chunks = []
    for chunk, id_val in zip(chunks, ids):
        if id_val not in unique_ids:
            unique_ids.add(id_val)
            unique_chunks.append(chunk)

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        collection_name=clean_filename(file_name),
        embedding=embedding_function,
        ids=list(unique_ids),
        persist_directory=vector_store_path
    )
    return vectorstore

def create_vectorstore_from_texts(documents: list, api_key: str, file_name: str) -> Chroma:
    """Split the documents and create a vector store using embeddings."""
    docs = split_document(documents)
    embedding_function = get_embedding_function(api_key)
    return create_vectorstore(docs, embedding_function, file_name)

def load_vectorstore(file_name: str, api_key: str, vectorstore_path: str = "db") -> Chroma:
    """Load an existing vector store using the file name and API key."""
    embedding_function = get_embedding_function(api_key)
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function,
        collection_name=clean_filename(file_name)
    )

def format_docs(docs: list) -> str:
    """Combine the text content of a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# =============================================================================
# Lesson Plan Generation Function
# =============================================================================

def generate_lesson_plan(client: OpenAI, grade: str, days: int, vectorstore: Chroma, topic: str) -> str:
    """
    Generate a lesson plan for the specified grade and number of days by retrieving
    context from the vector store and then querying the OpenAI API.
    """
    query = f"Information about topic '{topic}' and subtopic for {grade} grade"
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    context_text = format_docs(retrieved_docs)

    prompt = f"""
Create a comprehensive, detailed, and practical {days}-day lesson plan for a {grade} class on the topic '{topic}' with a focus on the subtopic from the unit. The course should be designed for a 50-minute class period each day, totaling minutes over {days} day(s).

Use the following context to inform your lesson plan:
{context_text}

Please structure the lesson plan with the following sections:

**1. Purpose**
- Explain the main focus of the lesson and its relationship to the curriculum.
- Outline relevant content standards and performance standards.
- Describe how this connects to real-world applications.

**2. Objectives**
- List 3-4 specific, measurable learning objectives starting with "By the end of this lesson, students will be able to:"
- Ensure objectives align with the content standards.
- Include both knowledge and skill-based objectives.

**3. Planning and Preparation Notes**
- Materials and Resources needed.
- Classroom Setup requirements.
- Lesson Timing breakdown:
  - Introduction:
  - Mini Lesson:
  - Guided Practice:
  - Independent Practice:
  - Assessment:
- Anticipated Challenges and solutions.
- Differentiation strategies for various learning levels.

**4. Prior Knowledge**
- What Students Should Already Know.
- How to Evaluate Prior Knowledge.
- Teaching at The Right Level strategies.
- Approaches for different student levels (struggling, comfortable, advanced).

**5. Lesson Flow**
Provide a detailed breakdown for each day of the lesson plan. For every day, include the following segments:

- **Introduction:**
  - Hook and engagement strategy.
- **Mini-Lesson:**
  - Core concept presentation.
- **Guided Practice:**
  - Group activities and teacher support.
- **Syllabus Breakdown (Daily):**
  - Present a detailed syllabus for the day that covers all relevant topics and subtopics, with emphasis on the subtopic.
  - Include specific subtopic details such as definitions, examples, historical context, real-world applications, prerequisites, and interconnections between topics.
  - Ensure the daily syllabus connects with the overall curriculum structure and suggests recommended supplemental resources.
- **Independent Practice:**
  - Individual tasks.
- **Assessment and Wrap-Up:**
  - Checking understanding and closure.

**6. Extension/Enrichment**
- Additional activities for advanced learners.
- Alternative approaches for deeper understanding.
- Creative projects or applications.
- Take-home activities.

**7. Assessment Tools**
- Diagnostic Assessments (pre-lesson evaluation).
- Formative Assessments (during-lesson checks).
- Summative Assessments (post-lesson evaluation).
- Assessment notes for teachers.
- Differentiation strategies for assessment.
- Feedback mechanisms.

Ensure the plan is practical, thorough, and includes clear instructions for implementation. Use specific examples and provide detailed guidance for teachers to effectively deliver the lesson.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an experienced educator who creates detailed, practical lesson plans."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error generating lesson plan: {str(e)}")
