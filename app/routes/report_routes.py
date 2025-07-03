from flask import Blueprint, request, jsonify, render_template
from app.models.database import db
from app.services.dataset_service_simple import DatasetServiceSimple
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import json
import numpy as np

logger = logging.getLogger(__name__)

report_bp = Blueprint('reports', __name__)

@report_bp.route('/reports/dashboard', methods=['GET'])
def dashboard_report():
    """Generate a comprehensive dashboard report."""
    try:
        # Get all datasets
        datasets = DatasetServiceSimple.list_datasets()
        
        # Calculate summary statistics
        total_datasets = len(datasets)
        total_records = sum(d.get('n_total_records', 0) for d in datasets)
        
        # Calculate revenue metrics (if purchase records exist)
        revenue_data = _calculate_revenue_metrics()
        
        # Get recent activity
        recent_activity = _get_recent_activity()
        
        # Calculate privacy metrics
        privacy_metrics = _calculate_privacy_metrics(datasets)
        
        # Get top performing datasets
        top_datasets = _get_top_performing_datasets(datasets)
        
        report_data = {
            'summary': {
                'total_datasets': total_datasets,
                'total_records': total_records,
                'total_revenue': revenue_data.get('total_revenue', 0),
                'avg_price_per_dataset': revenue_data.get('avg_price_per_dataset', 0),
                'total_queries': revenue_data.get('total_queries', 0)
            },
            'revenue_metrics': revenue_data,
            'recent_activity': recent_activity,
            'privacy_metrics': privacy_metrics,
            'top_datasets': top_datasets,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Error generating dashboard report: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/dataset/<dataset_id>', methods=['GET'])
def dataset_report(dataset_id):
    """Generate a detailed report for a specific dataset."""
    try:
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Get dataset statistics
        dataset_stats = _calculate_dataset_statistics(dataset_id, dataset)
        
        # Get pricing analysis
        pricing_analysis = _analyze_dataset_pricing(dataset)
        
        # Get privacy analysis
        privacy_analysis = _analyze_dataset_privacy(dataset)
        
        # Get usage statistics
        usage_stats = _get_dataset_usage_stats(dataset_id)
        
        report_data = {
            'dataset_info': {
                'dataset_id': dataset_id,
                'name': dataset.get('dataset_name'),
                'description': dataset.get('description'),
                'created_at': dataset.get('created_at'),
                'sensitive_attribute': dataset.get('sensitive_attribute_name'),
                'total_records': dataset.get('n_total_records'),
                'total_attributes': len(dataset.get('all_attribute_names', [])),
                'all_attributes': dataset.get('all_attribute_names', [])
            },
            'statistics': dataset_stats,
            'pricing_analysis': pricing_analysis,
            'privacy_analysis': privacy_analysis,
            'usage_statistics': usage_stats,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Error generating dataset report for {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/revenue', methods=['GET'])
def revenue_report():
    """Generate a detailed revenue report."""
    try:
        # Get date range from query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Calculate revenue metrics
        revenue_data = _calculate_revenue_metrics(start_date, end_date)
        
        # Get revenue by dataset
        revenue_by_dataset = _get_revenue_by_dataset(start_date, end_date)
        
        # Get revenue trends
        revenue_trends = _get_revenue_trends(start_date, end_date)
        
        # Get pricing analysis
        pricing_analysis = _analyze_pricing_performance()
        
        report_data = {
            'summary': revenue_data,
            'revenue_by_dataset': revenue_by_dataset,
            'revenue_trends': revenue_trends,
            'pricing_analysis': pricing_analysis,
            'date_range': {
                'start_date': start_date,
                'end_date': end_date
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Error generating revenue report: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/privacy', methods=['GET'])
def privacy_report():
    """Generate a privacy analysis report."""
    try:
        datasets = DatasetServiceSimple.list_datasets()
        
        # Analyze privacy across all datasets
        privacy_analysis = _analyze_overall_privacy(datasets)
        
        # Get privacy metrics by dataset
        privacy_by_dataset = _get_privacy_by_dataset(datasets)
        
        # Get epsilon distribution
        epsilon_distribution = _get_epsilon_distribution()
        
        # Get privacy vs utility analysis
        privacy_utility_analysis = _analyze_privacy_vs_utility()
        
        report_data = {
            'overall_privacy': privacy_analysis,
            'privacy_by_dataset': privacy_by_dataset,
            'epsilon_distribution': epsilon_distribution,
            'privacy_utility_analysis': privacy_utility_analysis,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Error generating privacy report: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/usage', methods=['GET'])
def usage_report():
    """Generate a usage analytics report."""
    try:
        # Get date range from query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Get usage statistics
        usage_stats = _get_overall_usage_stats(start_date, end_date)
        
        # Get popular attributes
        popular_attributes = _get_popular_attributes(start_date, end_date)
        
        # Get query patterns
        query_patterns = _analyze_query_patterns(start_date, end_date)
        
        # Get user behavior
        user_behavior = _analyze_user_behavior(start_date, end_date)
        
        report_data = {
            'usage_statistics': usage_stats,
            'popular_attributes': popular_attributes,
            'query_patterns': query_patterns,
            'user_behavior': user_behavior,
            'date_range': {
                'start_date': start_date,
                'end_date': end_date
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Error generating usage report: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/datasets', methods=['GET'])
def datasets_list():
    """Get list of all datasets for dropdowns and selection."""
    try:
        datasets = DatasetServiceSimple.list_datasets()
        
        # Format datasets for frontend consumption
        formatted_datasets = []
        for dataset in datasets:
            dataset_name = dataset.get('dataset_name') or dataset.get('name') or 'Unknown Dataset'
            formatted_datasets.append({
                'dataset_id': dataset.get('dataset_id'),
                'name': dataset_name,
                'description': dataset.get('description', ''),
                'total_records': dataset.get('n_total_records', 0),
                'total_attributes': len(dataset.get('all_attribute_names', [])),
                'sensitive_attribute': dataset.get('sensitive_attribute_name'),
                'created_at': dataset.get('created_at'),
                'all_attributes': dataset.get('all_attribute_names', [])
            })
        
        return jsonify(formatted_datasets), 200
        
    except Exception as e:
        logger.error(f"Error getting datasets list: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/model-comparison', methods=['GET'])
def model_comparison_report():
    """Generate a model comparison report showing the impact of differential privacy on ML performance."""
    try:
        # Get parameters from query string
        dataset_id = request.args.get('dataset_id')
        target_column = request.args.get('target_column')
        epsilon_values = request.args.get('epsilon_values', '0.1,0.5,1.0')
        
        # Validate and parse test_size
        try:
            test_size = float(request.args.get('test_size', '0.2'))
            if test_size > 1:
                test_size = test_size / 100.0
            if test_size <= 0 or test_size >= 1:
                return jsonify({'error': 'test_size must be between 0 and 1 (e.g., 0.2 for 20% test data, or 20 for 20%)'}), 400
        except ValueError:
            return jsonify({'error': 'test_size must be a valid number between 0 and 1 (e.g., 0.2 for 20% test data, or 20 for 20%)'}), 400
        
        if not dataset_id:
            return jsonify({'error': 'dataset_id is required'}), 400
        
        # Parse epsilon values
        try:
            epsilon_list = [float(x.strip()) for x in epsilon_values.split(',')]
        except ValueError:
            epsilon_list = [0.1, 0.5, 1.0]
        
        # Get dataset info
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Get available columns
        all_attributes = dataset.get('all_attribute_names', [])
        
        # If target_column not specified, try to use sensitive attribute
        if not target_column:
            target_column = dataset.get('sensitive_attribute_name')
            if not target_column or target_column not in all_attributes:
                # Use the first numeric column as default
                target_column = all_attributes[0] if all_attributes else None
        
        if not target_column:
            return jsonify({'error': 'No suitable target column found'}), 400
        
        # Get feature columns (all except target)
        feature_columns = [attr for attr in all_attributes if attr != target_column]
        
        # Calculate appropriate max_samples based on test_size to ensure sufficient test data
        # For a test_size of 0.2, we want at least 1000 test samples, so max_samples = 1000/0.2 = 5000
        # For larger datasets, we can use more samples
        min_test_samples = 1000
        calculated_max_samples = int(min_test_samples / test_size)
        
        # Cap at a reasonable maximum to avoid performance issues
        max_samples = min(calculated_max_samples, 50000)
        
        # Prepare the request for the analytics endpoint
        analytics_request = {
            'dataset_id': dataset_id,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'epsilon_values': epsilon_list,
            'test_size': test_size,
            'max_samples': max_samples
        }
        
        # Call the analytics endpoint
        import requests
        import json
        
        try:
            # Make internal request to analytics endpoint
            from flask import current_app
            with current_app.test_client() as client:
                response = client.post('/analytics/model-comparison', 
                                    json=analytics_request,
                                    content_type='application/json')
                
                if response.status_code != 200:
                    return jsonify({'error': 'Failed to generate model comparison'}), 500
                
                comparison_data = response.get_json()
                
        except Exception as e:
            logger.error(f"Error calling analytics endpoint: {e}")
            return jsonify({'error': 'Failed to generate model comparison'}), 500
        
        # Format the report data
        report_data = {
            'dataset_info': {
                'dataset_id': dataset_id,
                'name': dataset.get('dataset_name'),
                'target_column': target_column,
                'feature_columns': feature_columns,
                'total_attributes': len(all_attributes),
                'data_size': comparison_data.get('original', {}).get('data_size', 0)
            },
            'model_comparison': comparison_data,
            'recommendations': generate_model_comparison_recommendations(comparison_data),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return jsonify(report_data), 200
        
    except Exception as e:
        logger.error(f"Error generating model comparison report: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/purchase-query/<query_id>', methods=['GET'])
def purchase_query_by_id(query_id):
    """Return details of a purchased query by query_id."""
    try:
        if not db.is_connected():
            return jsonify({'error': 'Database not connected'}), 503
            
        query_doc = db.queries_collection.find_one({"query_id": query_id})
        if not query_doc:
            return jsonify({'error': 'Query ID not found'}), 404
        
        # Convert ObjectId to string for JSON serialization
        def convert_objectid(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        convert_objectid(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                convert_objectid(item)
                    elif hasattr(value, '__class__') and value.__class__.__name__ == 'ObjectId':
                        obj[key] = str(value)
            return obj
        
        # Convert any ObjectId fields to strings
        query_doc = convert_objectid(query_doc)
            
        return jsonify(query_doc), 200
    except Exception as e:
        logger.error(f"Error fetching purchase by query_id {query_id}: {e}")
        return jsonify({'error': str(e)}), 500

@report_bp.route('/reports/data-consumer', methods=['GET'])
def data_consumer_page():
    """Render the data consumer interface page."""
    return render_template('data_consumer_report.html')

@report_bp.route('/reports/data-consumer/query', methods=['POST'])
def data_consumer_query():
    """Execute a query on both original and differentially private data."""
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        filters = data.get('filters', [])
        
        if not dataset_id:
            return jsonify({'error': 'Dataset ID is required'}), 400
        
        # Get dataset info
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        all_attributes = dataset.get('all_attribute_names', [])
        
        # Query original data
        original_records = DatasetServiceSimple.get_dataset_records(
            dataset_id=dataset_id,
            selected_attributes=all_attributes,
            filters=filters,
            # limit=10000
        )
        original_count = len(original_records)
        
        # Query differentially private data for all epsilon values
        epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        dp_counts = {}
        
        for epsilon in epsilon_values:
            try:
                # Apply the same filters to DP data
                dp_records = _query_dp_records(dataset_id, epsilon, filters)
                dp_counts[epsilon] = len(dp_records)
            except Exception as e:
                logger.warning(f"Failed to query DP data for epsilon {epsilon}: {e}")
                dp_counts[epsilon] = 0
        
        # Prepare results for visualization
        results = {
            'original_count': original_count,
            'dp_counts': dp_counts,
            'epsilon_values': epsilon_values,
            'dataset_info': {
                'dataset_id': dataset_id,
                'name': dataset.get('dataset_name'),
                'total_records': dataset.get('n_total_records', 0)
            },
            'filters_applied': filters
        }

        # Patch: Add data_response for frontend count display (epsilon=0.5 by default)
        results['data_response'] = {
            'privatized_count': dp_counts.get(0.5, 0),
            'raw_count': original_count,
            'privatized_data': []  # Not changing any method, so leave empty
        }
        
        return jsonify(results), 200
        
    except Exception as e:
        logger.error(f"Error in data consumer query: {e}")
        return jsonify({'error': str(e)}), 500

def _query_dp_records(dataset_id, epsilon, filters):
    """Query differentially private records with filters."""
    if not db.is_connected():
        raise Exception("Database connection not available")
    
    # Build match conditions for filters
    match_conditions = {"dataset_id": dataset_id, "epsilon": epsilon}
    
    if filters:
        for filter_item in filters:
            attr = filter_item.get('attr')
            op = filter_item.get('op')
            val = filter_item.get('value')
            
            if attr and op and val is not None:
                field_path = f"data.{attr}"
                
                if op == '=':
                    match_conditions[field_path] = val
                elif op == '!=':
                    match_conditions[field_path] = {"$ne": val}
                elif op == '>':
                    match_conditions[field_path] = {"$gt": val}
                elif op == '>=':
                    match_conditions[field_path] = {"$gte": val}
                elif op == '<':
                    match_conditions[field_path] = {"$lt": val}
                elif op == '<=':
                    match_conditions[field_path] = {"$lte": val}
                elif op == 'contains':
                    match_conditions[field_path] = {"$regex": val, "$options": "i"}
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!",match_conditions)
    # Query the DP records collection
    dp_records = list(db.dataset_records_dp.find(match_conditions))
    return dp_records

# Helper functions for report generation

def _calculate_revenue_metrics(start_date=None, end_date=None):
    """Calculate revenue metrics for the specified date range."""
    try:
        # This would typically query a purchases/transactions collection
        # For now, return mock data
        return {
            'total_revenue': 12500.50,
            'total_queries': 45,
            'avg_price_per_query': 277.79,
            'avg_price_per_dataset': 3125.13,
            'revenue_growth': 15.2,
            'top_revenue_month': '2024-01'
        }
    except Exception as e:
        logger.error(f"Error calculating revenue metrics: {e}")
        return {}

def _get_recent_activity():
    """Get recent activity across the platform."""
    try:
        # This would query recent transactions, uploads, etc.
        return [
            {
                'type': 'dataset_upload',
                'dataset_name': 'Adult Income Data',
                'timestamp': datetime.utcnow().isoformat(),
                'user': 'producer_001'
            },
            {
                'type': 'data_purchase',
                'dataset_name': 'Bank Marketing Data',
                'amount': 450.00,
                'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'user': 'consumer_002'
            }
        ]
    except Exception as e:
        logger.error(f"Error getting recent activity: {e}")
        return []

def _calculate_privacy_metrics(datasets):
    """Calculate privacy metrics across all datasets."""
    try:
        total_datasets = len(datasets)
        datasets_with_privacy = sum(1 for d in datasets if d.get('sensitive_attribute_name'))
        
        return {
            'total_datasets': total_datasets,
            'datasets_with_privacy': datasets_with_privacy,
            'privacy_coverage': (datasets_with_privacy / total_datasets * 100) if total_datasets > 0 else 0,
            'avg_epsilon': 0.5,  # Mock average epsilon
            'privacy_levels': {
                'high': 15,
                'medium': 25,
                'low': 10
            }
        }
    except Exception as e:
        logger.error(f"Error calculating privacy metrics: {e}")
        return {}

def _get_top_performing_datasets(datasets):
    """Get top performing datasets based on various metrics."""
    try:
        # Sort datasets by number of records (as a proxy for performance)
        sorted_datasets = sorted(datasets, key=lambda x: x.get('n_total_records', 0), reverse=True)
        
        return [
            {
                'dataset_id': d.get('dataset_id'),
                'name': d.get('dataset_name'),
                'records': d.get('n_total_records'),
                'revenue': 1250.00,  # Mock revenue
                'queries': 12  # Mock query count
            }
            for d in sorted_datasets[:5]
        ]
    except Exception as e:
        logger.error(f"Error getting top performing datasets: {e}")
        return []

def _calculate_dataset_statistics(dataset_id, dataset):
    """Calculate detailed statistics for a specific dataset."""
    try:
        return {
            'total_records': dataset.get('n_total_records', 0),
            'total_attributes': len(dataset.get('all_attribute_names', [])),
            'quasi_identifiers': len(dataset.get('quasi_identifiers', [])),
            'attribute_distribution': _get_attribute_distribution(dataset),
            'data_quality': _assess_data_quality(dataset_id),
            'privacy_score': _calculate_privacy_score(dataset)
        }
    except Exception as e:
        logger.error(f"Error calculating dataset statistics: {e}")
        return {}

def _analyze_dataset_pricing(dataset):
    """Analyze pricing for a specific dataset."""
    try:
        weights = dataset.get('weights_percentage', {})
        prices = dataset.get('prices_per_attribute', {})
        
        return {
            'initial_pricing': {
                'min': dataset.get('initial_p_min'),
                'max': dataset.get('initial_p_max')
            },
            'attribute_weights': weights,
            'price_distribution': prices,
            'avg_price_per_attribute': sum(prices.get('max', {}).values()) / len(prices.get('max', {})) if prices.get('max') else 0,
            'pricing_efficiency': _calculate_pricing_efficiency(weights, prices)
        }
    except Exception as e:
        logger.error(f"Error analyzing dataset pricing: {e}")
        return {}

def _analyze_dataset_privacy(dataset):
    """Analyze privacy aspects of a dataset."""
    try:
        sensitive_attr = dataset.get('sensitive_attribute_name')
        quasi_ids = dataset.get('quasi_identifiers', [])
        
        return {
            'sensitive_attribute': sensitive_attr,
            'quasi_identifiers': quasi_ids,
            'privacy_risk_score': _calculate_privacy_risk(quasi_ids, sensitive_attr),
            'anonymization_potential': _assess_anonymization_potential(dataset),
            'recommended_epsilon_range': _recommend_epsilon_range(dataset)
        }
    except Exception as e:
        logger.error(f"Error analyzing dataset privacy: {e}")
        return {}

def _get_dataset_usage_stats(dataset_id):
    """Get usage statistics for a specific dataset."""
    try:
        # This would query usage logs/transactions
        return {
            'total_queries': 15,
            'total_revenue': 2250.00,
            'avg_epsilon_used': 0.6,
            'popular_attributes': ['age', 'income', 'education'],
            'query_frequency': {
                'daily': 2,
                'weekly': 8,
                'monthly': 15
            }
        }
    except Exception as e:
        logger.error(f"Error getting dataset usage stats: {e}")
        return {}

# Additional helper functions (implemented as needed)

def _get_attribute_distribution(dataset):
    """Get distribution of attribute types in a dataset."""
    try:
        attribute_types = dataset.get('attribute_types', {})
        distribution = {}
        for attr_type in attribute_types.values():
            distribution[attr_type] = distribution.get(attr_type, 0) + 1
        return distribution
    except Exception as e:
        logger.error(f"Error getting attribute distribution: {e}")
        return {}

def _assess_data_quality(dataset_id):
    """Assess data quality for a dataset."""
    try:
        # This would analyze the actual data
        return {
            'completeness': 0.95,
            'consistency': 0.88,
            'accuracy': 0.92,
            'overall_score': 0.92
        }
    except Exception as e:
        logger.error(f"Error assessing data quality: {e}")
        return {}

def _calculate_privacy_score(dataset):
    """Calculate a privacy score for a dataset."""
    try:
        # Simple scoring based on number of quasi-identifiers
        quasi_ids = len(dataset.get('quasi_identifiers', []))
        total_attrs = len(dataset.get('all_attribute_names', []))
        
        if total_attrs == 0:
            return 0
        
        # More quasi-identifiers = lower privacy score
        privacy_score = max(0, 100 - (quasi_ids / total_attrs * 100))
        return round(privacy_score, 2)
    except Exception as e:
        logger.error(f"Error calculating privacy score: {e}")
        return 0

def _calculate_pricing_efficiency(weights, prices):
    """Calculate pricing efficiency based on weights and prices."""
    try:
        if not weights or not prices.get('max'):
            return 0
        
        # Calculate correlation between weights and prices
        weight_values = list(weights.values())
        price_values = list(prices['max'].values())
        
        if len(weight_values) != len(price_values):
            return 0
        
        # Simple efficiency metric
        total_weight = sum(weight_values)
        total_price = sum(price_values)
        
        if total_weight == 0 or total_price == 0:
            return 0
        
        efficiency = (total_price / total_weight) * 100
        return round(efficiency, 2)
    except Exception as e:
        logger.error(f"Error calculating pricing efficiency: {e}")
        return 0

def _calculate_privacy_risk(quasi_ids, sensitive_attr):
    """Calculate privacy risk based on quasi-identifiers and sensitive attribute."""
    try:
        risk_score = 0
        
        # More quasi-identifiers = higher risk
        risk_score += len(quasi_ids) * 10
        
        # Sensitive attribute type affects risk
        if sensitive_attr and any(keyword in sensitive_attr.lower() for keyword in ['income', 'salary', 'money', 'financial']):
            risk_score += 20
        
        return min(100, risk_score)
    except Exception as e:
        logger.error(f"Error calculating privacy risk: {e}")
        return 0

def _assess_anonymization_potential(dataset):
    """Assess the potential for anonymization."""
    try:
        quasi_ids = dataset.get('quasi_identifiers', [])
        attribute_types = dataset.get('attribute_types', {})
        
        potential = 100
        
        # Reduce potential based on number of quasi-identifiers
        potential -= len(quasi_ids) * 5
        
        # Reduce potential based on attribute types
        categorical_count = sum(1 for t in attribute_types.values() if t == 'categorical')
        potential -= categorical_count * 3
        
        return max(0, potential)
    except Exception as e:
        logger.error(f"Error assessing anonymization potential: {e}")
        return 0

def _recommend_epsilon_range(dataset):
    """Recommend epsilon range for a dataset."""
    try:
        risk_score = _calculate_privacy_risk(
            dataset.get('quasi_identifiers', []),
            dataset.get('sensitive_attribute_name')
        )
        
        if risk_score > 70:
            return {'min': 0.1, 'max': 0.3, 'recommended': 0.2}
        elif risk_score > 40:
            return {'min': 0.2, 'max': 0.5, 'recommended': 0.35}
        else:
            return {'min': 0.3, 'max': 0.8, 'recommended': 0.6}
    except Exception as e:
        logger.error(f"Error recommending epsilon range: {e}")
        return {'min': 0.1, 'max': 1.0, 'recommended': 0.5}

# Additional helper functions for other reports

def _get_revenue_by_dataset(start_date=None, end_date=None):
    """Get revenue breakdown by dataset."""
    try:
        datasets = DatasetServiceSimple.list_datasets()
        return [
            {
                'dataset_id': d.get('dataset_id'),
                'name': d.get('dataset_name'),
                'revenue': 1250.00,  # Mock data
                'queries': 12,
                'avg_price': 104.17
            }
            for d in datasets
        ]
    except Exception as e:
        logger.error(f"Error getting revenue by dataset: {e}")
        return []

def _get_revenue_trends(start_date=None, end_date=None):
    """Get revenue trends over time."""
    try:
        # Mock trend data
        return [
            {'date': '2024-01', 'revenue': 3200.00, 'queries': 8},
            {'date': '2024-02', 'revenue': 3200.00, 'queries': 12},
            {'date': '2024-03', 'revenue': 5200.00, 'queries': 15}
        ]
    except Exception as e:
        logger.error(f"Error getting revenue trends: {e}")
        return []

def _analyze_pricing_performance():
    """Analyze pricing performance across datasets."""
    try:
        return {
            'avg_price_range': {'min': 50.00, 'max': 500.00},
            'price_elasticity': 0.75,
            'optimal_pricing_strategy': 'dynamic',
            'pricing_recommendations': [
                'Consider lowering prices for high-volume datasets',
                'Increase prices for datasets with high privacy requirements'
            ]
        }
    except Exception as e:
        logger.error(f"Error analyzing pricing performance: {e}")
        return {}

def _analyze_overall_privacy(datasets):
    """Analyze overall privacy across all datasets."""
    try:
        total_datasets = len(datasets)
        privacy_scores = [_calculate_privacy_score(d) for d in datasets]
        
        return {
            'avg_privacy_score': sum(privacy_scores) / len(privacy_scores) if privacy_scores else 0,
            'privacy_distribution': {
                'high': sum(1 for s in privacy_scores if s >= 80),
                'medium': sum(1 for s in privacy_scores if 50 <= s < 80),
                'low': sum(1 for s in privacy_scores if s < 50)
            },
            'total_datasets': total_datasets
        }
    except Exception as e:
        logger.error(f"Error analyzing overall privacy: {e}")
        return {}

def _get_privacy_by_dataset(datasets):
    """Get privacy analysis by dataset."""
    try:
        return [
            {
                'dataset_id': d.get('dataset_id'),
                'name': d.get('dataset_name'),
                'privacy_score': _calculate_privacy_score(d),
                'quasi_identifiers': len(d.get('quasi_identifiers', [])),
                'sensitive_attribute': d.get('sensitive_attribute_name')
            }
            for d in datasets
        ]
    except Exception as e:
        logger.error(f"Error getting privacy by dataset: {e}")
        return []

def _get_epsilon_distribution():
    """Get distribution of epsilon values used."""
    try:
        return {
            'low_privacy': {'count': 15, 'percentage': 33.3},
            'medium_privacy': {'count': 20, 'percentage': 44.4},
            'high_privacy': {'count': 10, 'percentage': 22.2}
        }
    except Exception as e:
        logger.error(f"Error getting epsilon distribution: {e}")
        return {}

def _analyze_privacy_vs_utility():
    """Analyze the trade-off between privacy and utility."""
    try:
        return {
            'correlation': -0.75,
            'optimal_epsilon': 0.5,
            'trade_off_analysis': {
                'high_privacy_low_utility': 10,
                'balanced': 25,
                'low_privacy_high_utility': 10
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing privacy vs utility: {e}")
        return {}

def _get_overall_usage_stats(start_date=None, end_date=None):
    """Get overall usage statistics."""
    try:
        return {
            'total_queries': 45,
            'unique_users': 12,
            'avg_queries_per_user': 3.75,
            'peak_usage_hours': [10, 14, 16],
            'most_active_datasets': ['adult_income', 'bank_marketing']
        }
    except Exception as e:
        logger.error(f"Error getting overall usage stats: {e}")
        return {}

def _get_popular_attributes(start_date=None, end_date=None):
    """Get most popular attributes queried."""
    try:
        return [
            {'attribute': 'age', 'queries': 25, 'percentage': 55.6},
            {'attribute': 'income', 'queries': 20, 'percentage': 44.4},
            {'attribute': 'education', 'queries': 18, 'percentage': 40.0},
            {'attribute': 'occupation', 'queries': 15, 'percentage': 33.3},
            {'attribute': 'marital_status', 'queries': 12, 'percentage': 26.7}
        ]
    except Exception as e:
        logger.error(f"Error getting popular attributes: {e}")
        return []

def _analyze_query_patterns(start_date=None, end_date=None):
    """Analyze query patterns."""
    try:
        return {
            'query_types': {
                'aggregation': 20,
                'filtering': 15,
                'grouping': 10
            },
            'time_patterns': {
                'morning': 15,
                'afternoon': 20,
                'evening': 10
            },
            'complexity_distribution': {
                'simple': 25,
                'moderate': 15,
                'complex': 5
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing query patterns: {e}")
        return {}

def _analyze_user_behavior(start_date=None, end_date=None):
    """Analyze user behavior patterns."""
    try:
        return {
            'user_segments': {
                'researchers': {'count': 5, 'avg_queries': 8},
                'business_analysts': {'count': 4, 'avg_queries': 6},
                'students': {'count': 3, 'avg_queries': 3}
            },
            'preferred_epsilon_ranges': {
                '0.1-0.3': 10,
                '0.3-0.6': 20,
                '0.6-1.0': 15
            },
            'session_duration': {
                'avg_minutes': 25,
                'median_minutes': 18
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing user behavior: {e}")
        return {}

def generate_model_comparison_recommendations(comparison_data):
    """Generate recommendations based on model comparison results."""
    recommendations = []
    
    original_metrics = comparison_data.get('original', {}).get('metrics', {})
    noisy_comparisons = comparison_data.get('noisy_comparisons', [])
    summary = comparison_data.get('summary', {})
    
    if not original_metrics or not noisy_comparisons:
        return ["Insufficient data for recommendations"]
    
    # Analyze performance degradation
    avg_degradation = summary.get('average_degradation', {})
    model_type = summary.get('model_type', 'unknown')
    
    # Check if degradation is acceptable
    if model_type == 'classification':
        accuracy_degradation = avg_degradation.get('accuracy', 0)
        if accuracy_degradation < 5:
            recommendations.append("✅ Excellent: Model trained on clean data performs well on noisy test data (< 5% degradation)")
        elif accuracy_degradation < 15:
            recommendations.append("⚠️ Acceptable: Moderate performance drop when testing on noisy data (5-15% degradation)")
        else:
            recommendations.append("❌ High Impact: Significant performance drop on noisy test data (> 15%). Consider data preprocessing or model robustness techniques.")
    
    elif model_type == 'regression':
        r2_degradation = avg_degradation.get('r2_score', 0)
        if r2_degradation < 10:
            recommendations.append("✅ Excellent: Model trained on clean data performs well on noisy test data (< 10% degradation)")
        elif r2_degradation < 25:
            recommendations.append("⚠️ Acceptable: Moderate performance drop when testing on noisy data (10-25% degradation)")
        else:
            recommendations.append("❌ High Impact: Significant performance drop on noisy test data (> 25%). Consider data preprocessing or model robustness techniques.")
    
    # Epsilon recommendations based on test data noise tolerance
    best_epsilon = None
    best_performance = float('inf')
    
    for comparison in noisy_comparisons:
        epsilon = comparison.get('epsilon', 0)
        degradation = comparison.get('performance_degradation', {})
        
        if model_type == 'classification':
            current_degradation = degradation.get('accuracy', 0)
        else:
            current_degradation = degradation.get('r2_score', 0)
        
        if current_degradation < best_performance:
            best_performance = current_degradation
            best_epsilon = epsilon
    
    if best_epsilon is not None:
        recommendations.append(f"🎯 Recommended Test Data Privacy Level: ε={best_epsilon} (lowest performance degradation on noisy test data)")
    
    # Privacy vs Utility trade-off for test data
    if len(noisy_comparisons) >= 2:
        low_epsilon = min(c.get('epsilon', 0) for c in noisy_comparisons)
        high_epsilon = max(c.get('epsilon', 0) for c in noisy_comparisons)
        
        low_degradation = next((c.get('performance_degradation', {}).get('accuracy', 0) 
                               for c in noisy_comparisons if c.get('epsilon') == low_epsilon), 0)
        high_degradation = next((c.get('performance_degradation', {}).get('accuracy', 0) 
                                for c in noisy_comparisons if c.get('epsilon') == high_epsilon), 0)
        
        if high_degradation - low_degradation > 10:
            recommendations.append("⚖️ Significant impact: Test data privacy noise significantly affects model performance. Consider data preprocessing or robust model training.")
        else:
            recommendations.append("✅ Good robustness: Model trained on clean data handles test data noise well with minimal performance loss.")
    
    # Model-specific recommendations
    original_model = comparison_data.get('original', {}).get('model_name', '')
    if 'Random Forest' in original_model:
        recommendations.append("🌳 Random Forest models show good robustness to test data noise due to their ensemble nature.")
    elif 'Logistic' in original_model or 'Linear' in original_model:
        recommendations.append("📈 Linear models may be more sensitive to test data noise. Consider ensemble methods or data preprocessing for better robustness.")
    
    # Additional recommendations for real-world scenarios
    if len(noisy_comparisons) > 0:
        recommendations.append("💡 This analysis shows how well your model generalizes to noisy/differentially private test data - important for real-world privacy-preserving applications.")
    
    return recommendations 