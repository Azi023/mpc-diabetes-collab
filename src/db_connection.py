# src/db_connection.py 
import os
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load .env from repo root reliably
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class Database:
    def __init__(self):
        self.connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        if not self.connection_string:
            print("❌ MONGODB_CONNECTION_STRING not set in .env")
            raise SystemExit(1)
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        """Establishes the connection to the MongoDB database."""
        try:
            self.client = MongoClient(self.connection_string)
            # Modern health check
            self.client.admin.command("ping")
            print("✅ MongoDB connection successful.")
            # Use your DB name from Compass screenshot
            self.db = self.client["SecureHealthDB"]
        except ConnectionFailure as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise SystemExit(1)

    def get_collection(self, collection_name="patients"):
        if self.db is None:
            print("❌ Database not connected.")
            return None
        return self.db[collection_name]

    def close(self):
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed.")

db_connection = Database()
