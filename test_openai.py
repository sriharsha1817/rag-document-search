import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"API Key loaded: {api_key[:10]}..." if api_key else "No API key found")

try:
    # Test embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    result = embeddings.embed_query("test")
    print("✅ OpenAI API key works!")
    print(f"Embedding dimension: {len(result)}")
except Exception as e:
    print(f"❌ API key failed: {e}")