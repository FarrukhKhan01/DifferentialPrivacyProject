from flask import Flask
from flask_cors import CORS
import logging
from config import config

def create_app(config_name='default'):
    """Application factory pattern for Flask app creation."""
    
    # Create Flask app
    app = Flask(__name__, 
                static_folder='../static',
                template_folder='../templates')
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize CORS
    CORS(app)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if app.config['DEBUG'] else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints (using simplified routes)
    from app.routes.api_routes_simple import api_bp
    from app.routes.web_routes import web_bp
    from app.routes.analytics_routes import analytics_bp
    from app.routes.report_routes import report_bp
    app.register_blueprint(api_bp)  # No prefix - routes will be at root level
    app.register_blueprint(web_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(report_bp)
    
    # Health check route
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'message': 'FSD Data Marketplace API is running'}
    
    return app
