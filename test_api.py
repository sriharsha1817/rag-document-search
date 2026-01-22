import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print(f"API Key: {api_key}")
print(f"API Key length: {len(api_key) if api_key else 0}")

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
    result = embeddings.embed_query("test")
    print("✅ API key works!")
except Exception as e:
    print(f"❌ API key failed: {e}")