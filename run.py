#!/usr/bin/env python3
"""
Main entry point for the FSD Data Marketplace application.
"""

import os
import sys
from app import create_app
from config import Config

def main():
    """Main application entry point."""
    
    # Get configuration from environment
    config_name = os.environ.get('FLASK_ENV', 'default')
    
    # Create Flask application
    app = create_app(config_name)
    
    # Get configuration
    config = Config()
    
    # Run the application
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )

if __name__ == '__main__':
    main() 