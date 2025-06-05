import os
from dotenv import load_dotenv
import anthropic
from pymongo import MongoClient

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    try:
        # Connect to MongoDB
        mongo_uri = os.getenv('MONGO_URI')
        client = MongoClient(mongo_uri)
        
        # Test the connection
        client.admin.command('ping')
        print("✅ MongoDB Connection Successful!")
        
        # Test database access
        db = client["myjournal"]
        journal_entries = db["JournalEntries"]
        print("✅ Successfully accessed journal database and collection")
        
    except Exception as e:
        print("❌ MongoDB Error:", str(e))

def test_anthropic_connection():
    try:
        # Test Anthropic API
        client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": "Say hello!"
            }]
        )
        print("✅ Anthropic Connection Successful!")
        print("Claude's response:", message.content)
        
    except Exception as e:
        print("❌ Anthropic Error:", str(e))

if __name__ == "__main__":
    print("\nTesting MongoDB Connection...")
    test_mongodb_connection()
    
    print("\nTesting Anthropic Connection...")
    test_anthropic_connection()