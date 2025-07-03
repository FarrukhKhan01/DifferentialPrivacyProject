from pymongo import MongoClient
from config import Config
import logging

logger = logging.getLogger(__name__)

class Database:
    """Database connection and collection management."""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.client = None
        self.db = None
        self.datasets_collection = None
        self.dataset_records_collection = None  # New collection for individual records
        self.queries_collection = None
        self.dataset_records_dp = None  # Add this line for DP records
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.config.MONGO_URI)
            self.db = self.client[self.config.MONGO_DB_NAME]
            self.datasets_collection = self.db['datasets']
            self.dataset_records_collection = self.db['dataset_records']  # New collection
            self.queries_collection = self.db['queries']
            self.dataset_records_dp = self.db['dataset_records_dp']  # Add this line for DP records
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.datasets_collection = None
            self.dataset_records_collection = None
            self.queries_collection = None
            self.dataset_records_dp = None
    
    def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # Index on dataset_id for fast lookups
            try:
                self.dataset_records_collection.create_index("dataset_id")
            except Exception as e:
                if "already exists" not in str(e).lower() and "IndexKeySpecsConflict" not in str(e):
                    logger.warning(f"Could not create dataset_id index: {e}")
            
            # Compound index for efficient filtering
            try:
                self.dataset_records_collection.create_index([
                    ("dataset_id", 1),
                    ("record_id", 1)
                ])
            except Exception as e:
                if "already exists" not in str(e).lower() and "IndexKeySpecsConflict" not in str(e):
                    logger.warning(f"Could not create compound index: {e}")
            
            # Index for queries collection
            try:
                self.queries_collection.create_index("query_id")
            except Exception as e:
                if "already exists" not in str(e).lower() and "IndexKeySpecsConflict" not in str(e):
                    logger.warning(f"Could not create query_id index: {e}")
                    
            try:
                self.queries_collection.create_index("dataset_id")
            except Exception as e:
                if "already exists" not in str(e).lower() and "IndexKeySpecsConflict" not in str(e):
                    logger.warning(f"Could not create dataset_id index for queries: {e}")
            
            logger.info("Database indexes checked/created successfully.")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def is_connected(self):
        """Check if database connection is active."""
        return self.client is not None and self.db is not None
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

# Global database instance
db = Database() 