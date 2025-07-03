# Environment Configuration
# Copy this file to .env and modify as needed

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=your-secret-key-change-in-production

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=fsd_db

# API Configuration
API_HOST=localhost
API_PORT=5001

# Privacy Configuration
DEFAULT_DP_EPSILON_MIN=0.1
DEFAULT_DP_EPSILON_MAX=1.0
DEFAULT_DP_DELTA=1e-5 