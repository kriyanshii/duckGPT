import streamlit as st
import os
from pymongo import MongoClient
from google.api_core.exceptions import NotFound, RetryError
from google.generativeai import list_models
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import ServiceContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext, VectorStoreIndex
import time

def list_available_models():
    try:
        models = list_models()
        for model in models:
            print(f"Model Name: {model.name}, Display Name: {model.display_name}")
    except NotFound as e:
        print(f"Error listing models: {e}")

def initialize_gemini_models(retry_attempts=5, initial_delay=1):
    for attempt in range(retry_attempts):
        try:
            embed_model = GeminiEmbedding(model_name="models/embedding-001")
            print("Embedding model initialized")
            llm = Gemini(model="models/gemini-1.5-pro")  # Ensure this is a valid model name
            print("Gemini model initialized successfully")
            return embed_model, llm
        except RetryError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retry_attempts - 1:
                time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff
            else:
                raise
    return None, None

# web app config, configure the streamlit app as you'd like
st.set_page_config(page_title="AI Tutor", layout="wide", page_icon="🦆")
st.title("AI Tutor Using Feynman Technique")

# load credentials from the .streamlit/secrets.toml file
secrets = st.secrets

print("secrets are set")

# Get and assign each of the environment variables
os.environ["GOOGLE_API_KEY"] = secrets["GEMINI_API_KEY"]
ATLAS_URI = secrets["ATLAS_URI"]
DB_NAME = secrets["DB_NAME"]
COLLECTION_NAME = secrets["COLLECTION_NAME"]
INDEX_NAME = secrets["INDEX_NAME"]
print("envs are set")


# set up mongodb client with SSL/TLS configuration
try:
    mongodb_client = MongoClient(ATLAS_URI, tls=True, tlsAllowInvalidCertificates=True)
    # Attempt to connect to check if the credentials are correct
    mongodb_client.admin.command('ping')
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")
    st.stop()
print("connection to mongodb is set")

# List available models to verify the correct model name
list_available_models()

# Use a valid model name obtained from the listing
valid_model_name = "models/embedding-001"  # Replace with actual valid model name

# Initialize gemini models with retry logic
try:
    embed_model, llm = initialize_gemini_models()
except RetryError as e:
    st.error(f"Failed to initialize Gemini models after multiple attempts: {e}")
    st.stop()

if embed_model and llm:
    # create llama_index service context
    Settings.embed_model = embed_model
    Settings.llm = llm

    # set up the vector store with the details, to have access to the specific data
    vector_store = MongoDBAtlasVectorSearch(mongodb_client=mongodb_client,
                                            db_name=DB_NAME, collection_name=COLLECTION_NAME,
                                            index_name=INDEX_NAME)

    # Create the vector store and index and the pipeline for vector search will be created
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # set up the created index as a query engine
    query_llm = index.as_query_engine()

    # chat interface for consistent queries
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Function to display messages
    def display_messages():
        for message, kind in st.session_state.messages:
            with st.chat_message(kind):
                st.markdown(message)

    # Function to handle user input
    def handle_user_input(prompt):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append((prompt, "user"))

        # using the query engine to get response, rendering the answer and adding to conversation history
        with st.spinner("Generating response..."):
            try:
                answer = query_llm.query(prompt)
                if answer:
                    st.chat_message("ai").markdown(answer)
                    st.session_state.messages.append((answer, "ai"))
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Display for all the messages
    display_messages()

    # Text input from the user
    user_input = st.text_input("Enter your topic or question:")

    # Button to submit the text input
    if st.button("Submit"):
        if user_input:
            handle_user_input(user_input)

    # Voice input (requires additional configuration for voice recognition)
    st.write("Voice input is currently not supported. Please use text input.")

else:
    st.error("Failed to initialize Gemini models. Please check your configuration and try again.")
