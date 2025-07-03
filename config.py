import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the FSD Data Marketplace application."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # MongoDB Configuration
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/'
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME') or 'fsd_db'
    
    # API Configuration
    API_HOST = os.environ.get('API_HOST') or 'localhost'
    API_PORT = int(os.environ.get('API_PORT') or 5001)
    
    # Data Processing Configuration
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_CSV_ENCODINGS = ['utf-8', 'latin-1', 'cp1252']
    
    # Privacy Configuration
    DEFAULT_DP_EPSILON_MIN = 0.1
    DEFAULT_DP_EPSILON_MAX = 1.0
    DEFAULT_DP_DELTA = 1e-5
    
    # Pricing Configuration
    DEFAULT_P_MAX = 10000
    DEFAULT_P_MIN = 1000
    
    # File Paths
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    TEMPLATES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    MONGO_DB_NAME = 'fsd_test_db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 