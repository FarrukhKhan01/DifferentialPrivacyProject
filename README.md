# FSD Data Marketplace

A privacy-aware data marketplace platform that implements Fair Secure Data (FSD) principles with differential privacy.

## 🏗️ Project Structure

```
dataPrivacy/
├── app/                          # Main application package
│   ├── models/                   # Database models
│   │   ├── __init__.py
│   │   └── database.py          # MongoDB connection management
│   ├── routes/                   # Route definitions
│   │   ├── __init__.py
│   │   ├── api_routes.py        # REST API endpoints
│   │   └── web_routes.py        # Web interface routes
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   └── dataset_service.py   # Dataset operations
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── data_processing.py   # CSV processing & validation
│       ├── privacy.py           # Differential privacy & mutual information
│       └── pricing.py           # Price calculations
├── static/                       # Static files (CSS, JS, images)
├── templates/                    # HTML templates
├── uploads/                      # File uploads
├── data/                         # Data files
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── run.py                        # Application entry point
└── env_config.py                 # Environment configuration template
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MongoDB (running on localhost:27017)
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dataPrivacy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the environment template
   cp env_config.py .env
   # Edit .env with your configuration
   ```

5. **Start MongoDB**
   ```bash
   # Make sure MongoDB is running on localhost:27017
   mongod
   ```

6. **Run the application**
   ```bash
   python run.py
   ```

7. **Access the application**
   - Web Interface: http://localhost:5001
   - API Health Check: http://localhost:5001/health

## 📚 API Documentation

### Producer Endpoints

#### Create Dataset
```http
POST /api/producer/datasets
Content-Type: application/json

{
  "dataset_name": "Adult Income Data",
  "csv_content": "...",
  "initial_p_max": 10000,
  "initial_p_min": 1000,
  "sensitive_attribute_name": "income",
  "attribute_types": {
    "age": "numeric",
    "income": "categorical"
  },
  "description": "Adult income dataset",
  "numeric_attribute_ranges": {
    "age": [17, 90]
  }
}
```

#### Get Dataset Details
```http
GET /api/producer/datasets/{dataset_id}
```

### Consumer Endpoints

#### List Available Datasets
```http
GET /api/marketplace/datasets
```

#### Get Dataset Information
```http
GET /api/marketplace/datasets/{dataset_id}/info
```

#### Query Dataset Price
```http
POST /api/marketplace/datasets/{dataset_id}/query
Content-Type: application/json

{
  "selected_attributes": ["age", "income"],
  "num_values_k": 100,
  "dp_epsilon": 0.5
}
```

#### Purchase Dataset
```http
POST /api/marketplace/purchase
Content-Type: application/json

{
  "dataset_id": "uuid",
  "selected_attributes": ["age", "income"],
  "num_values_k": 100,
  "dp_epsilon": 0.5,
  "payment_amount": 5000
}
```

## 🔧 Configuration

The application uses environment variables for configuration. Key settings:

- `FLASK_ENV`: Environment (development/production/testing)
- `MONGO_URI`: MongoDB connection string
- `API_HOST`: Host to bind the server to
- `API_PORT`: Port to run the server on
- `SECRET_KEY`: Flask secret key

## 🛡️ Privacy Features

- **Differential Privacy**: Implements ε-differential privacy with configurable epsilon values
- **Mutual Information**: Calculates attribute weights based on mutual information with sensitive attributes
- **Fair Pricing**: Price calculation based on privacy level and data utility
- **Data Validation**: Comprehensive input validation and sanitization

## 🧪 Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/
```

## 📝 Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes

### Adding New Features
1. Create feature branch
2. Implement changes in appropriate modules
3. Add tests
4. Update documentation
5. Submit pull request

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team

## 🔄 Changelog

### v1.0.0
- Initial release
- Basic data marketplace functionality
- Differential privacy implementation
- Fair pricing model 