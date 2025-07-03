from flask import Blueprint, render_template

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Serve the main marketplace interface."""
    return render_template('index.html')

@web_bp.route('/producer')
def producer_portal():
    """Producer portal page."""
    return render_template('index.html')

@web_bp.route('/marketplace')
def consumer_marketplace():
    """Consumer marketplace page."""
    return render_template('index.html')

@web_bp.route('/consumer-simple')
def consumer_simplified():
    """Optimized consumer interface."""
    return render_template('consumer_optimized.html')

@web_bp.route('/reports')
def reports():
    """Reports page."""
    return render_template('reports.html')

def query_comparison():
    pass 