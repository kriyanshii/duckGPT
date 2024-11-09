# Load settings from .env file
import sys
import pymongo
import time
from dotenv import find_dotenv, dotenv_values
import os
from llama_index.embeddings.gemini import GeminiEmbedding
from google.api_core.exceptions import DeadlineExceeded


# Change system path to root direcotry
sys.path.insert(0, '../')

# _ = load_dotenv(find_dotenv()) # read local .env file
config = dotenv_values(find_dotenv())


ATLAS_URI = config.get('ATLAS_URI')
DB_NAME = config.get('DB_NAME')
COLLECTION_NAME = config.get('COLLECTION_NAME')
INDEX_NAME = config.get('INDEX_NAME')


if not ATLAS_URI:
    raise Exception ("'ATLAS_URI' is not set.  Please set it above to continue...")
else:
    print("ATLAS_URI Connection string found")


mongodb_client = pymongo.MongoClient(ATLAS_URI)
print ("Atlas client initialized")

db = mongodb_client[DB_NAME]
collection = db[COLLECTION_NAME]

print ("MongoDB connected successfully")

os.environ["GOOGLE_API_KEY"] = config.get("GEMINI_API_KEY")
print("Google API key set", os.environ["GOOGLE_API_KEY"])


embed_model = GeminiEmbedding(model_name="models/embedding-001", title="this is a document")
print("Embedding model initialized")

def generate_embedding_with_retry(embed_model, text, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Attempt to get the embedding with an increased timeout if supported
            return embed_model.get_text_embedding(text)
        except DeadlineExceeded as e:
            if attempt < max_retries - 1:
                print(f"Request timed out. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Max retries reached. Failed to generate embedding.")
                raise e

# Test the embedding generation with retry
try:
    text = "What is the meaning of life"
    vector = generate_embedding_with_retry(embed_model, text)
    print("Embedding generated successfully:", vector)
    print(f"The dimension of the embedding model is: {len(vector)}")
    
except DeadlineExceeded:
    print("Failed to generate embedding due to timeout.")

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
# service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
Settings.llm = None
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900


print("Service context initialized")

from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
# from llama_index.storage.storage_context import StorageContext
# Uncomment the line above and comment away the line below if you face an import error
from llama_index.core import StorageContext

vector_store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client,
                                 db_name = DB_NAME, collection_name = COLLECTION_NAME,
                                 vector_index_name = INDEX_NAME)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
