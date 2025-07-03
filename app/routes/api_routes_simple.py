from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from flask import Blueprint, request, jsonify, current_app
import logging
import csv
import io
from app.services.dataset_service_simple import DatasetServiceSimple
from app.models.database import db
import uuid
import numpy as np  # Added for Laplace noise

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

@api_bp.route('/producer/datasets', methods=['POST'])
def create_dataset_entry():
    """Create a new dataset entry."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['dataset_name', 'csv_content', 'initial_p_max', 'initial_p_min', 
                          'sensitive_attribute_name', 'attribute_types']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract data
        dataset_name = data['dataset_name']
        csv_content = data['csv_content']
        initial_p_max = float(data['initial_p_max'])
        initial_p_min = float(data['initial_p_min'])
        sensitive_attribute_name = data['sensitive_attribute_name']
        attribute_types = data['attribute_types']
        description = data.get('description', '')
        numeric_attribute_ranges = data.get('numeric_attribute_ranges', {})
        
        # Validate prices
        if initial_p_max <= initial_p_min:
            return jsonify({'error': 'initial_p_max must be greater than initial_p_min'}), 400
        
        # Create dataset
        result = DatasetServiceSimple.create_dataset(
            dataset_name=dataset_name,
            csv_content=csv_content,
            initial_p_max=initial_p_max,
            initial_p_min=initial_p_min,
            sensitive_attribute_name=sensitive_attribute_name,
            attribute_types=attribute_types,
            numeric_attribute_ranges=numeric_attribute_ranges,
            description=description
        )
        
        return jsonify(result), 201
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/producer/datasets/<string:dataset_id>', methods=['GET'])
def get_dataset_details_producer(dataset_id):
    """Get dataset details for producer."""
    try:
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        return jsonify(dataset), 200
        
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/datasets', methods=['GET'])
def list_available_datasets():
    """List all available datasets for consumers."""
    try:
        datasets = DatasetServiceSimple.list_datasets()
        return jsonify(datasets), 200  # Return just the array, not wrapped in object
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/datasets/<string:dataset_id>/info', methods=['GET'])
def get_dataset_info_for_consumer(dataset_id):
    """Get dataset information for consumer (without sensitive data)."""
    try:
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Return only public information in the format expected by frontend
        all_attributes = dataset.get('all_attribute_names', [])
        attribute_types = dataset.get('attribute_types', {})
        sensitive_attr = dataset.get('sensitive_attribute_name')
        
        # Create queryable attributes (all except sensitive)
        queryable_attributes = {}
        for attr in all_attributes:
            if attr != sensitive_attr:
                queryable_attributes[attr] = attribute_types.get(attr, 'unknown')
        
        # Handle both old (name) and new (dataset_name) field names
        dataset_name = dataset.get('dataset_name') or dataset.get('name') or 'Unknown Dataset'
        
        public_info = {
            'name': dataset_name,  # Frontend expects 'name'
            'description': dataset.get('description', ''),
            'total_records': dataset.get('n_total_records'),
            'queryable_attributes': queryable_attributes,
            'sensitive_attribute': sensitive_attr
        }
        
        return jsonify(public_info), 200
        
    except Exception as e:
        logger.error(f"Error getting dataset info {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500
    
@api_bp.route('/marketplace/datasets/<string:dataset_id>/query', methods=['POST'])
def query_dataset_price(dataset_id):
    """Calculate price for a dataset query, supporting filters."""
    try:
        data = request.get_json()
        required_fields = ['selected_attributes', 'num_values_per_attribute', 'epsilon']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        selected_attributes = data['selected_attributes']
        num_values_k = int(data['num_values_per_attribute'])
        dp_epsilon = float(data['epsilon'])
        filters = data.get('filters', [])
        if num_values_k <= 0:
            return jsonify({'error': 'num_values_per_attribute must be positive'}), 400
        if dp_epsilon < 0.1 or dp_epsilon > 1:
            return jsonify({'error': 'epsilon must be between 0.1 and 1'}), 400

        # Get dataset metadata
        dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        # Count filtered records for price calculation
        n_rows_for_price = DatasetServiceSimple.count_dataset_records(dataset_id, filters)
        
        # If fewer rows than requested, adjust k
        k_for_price = min(num_values_k, n_rows_for_price)
        
        # Calculate price based on filtered data
        price_info = DatasetServiceSimple.calculate_query_price(
            dataset_id, selected_attributes, k_for_price, dp_epsilon, filters
        )
        
        # Generate query_id for purchase flow
        query_id = str(uuid.uuid4())
        query_doc = {
            "query_id": query_id,
            "dataset_id": dataset_id,
            "query_details_from_consumer": data,
            "calculated_p_max_k_for_query": price_info["p_max_k_query"],
            "calculated_p_min_k_for_query": price_info["p_min_k_query"],
            "final_price_offered": price_info["final_price"],
            "dp_epsilon_for_price_and_privacy": dp_epsilon,
            "filtered_count": price_info.get("filtered_count", n_rows_for_price),
            "total_count": price_info.get("total_count", 0),
            "scarcity_factor": price_info.get("scarcity_factor", 1.0),
            "status": "price_offered",
            "query_timestamp": "2024-01-01T00:00:00Z"
        }
        if db.is_connected():
            db.queries_collection.insert_one(query_doc)
        return jsonify({
            "query_id": query_id,
            "final_price": price_info["final_price"],
            "p_max_k_for_query": price_info["p_max_k_query"],
            "p_min_k_for_query": price_info["p_min_k_query"],
            "dataset_id": dataset_id,
            "dp_epsilon_applied": dp_epsilon,
            "filtered_count": price_info.get("filtered_count", n_rows_for_price),
            "total_count": price_info.get("total_count", 0),
            "scarcity_factor": price_info.get("scarcity_factor", 1.0),
            "price_breakdown": price_info.get("price_breakdown", {})
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error calculating price for dataset {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/purchase', methods=['POST'])
def purchase_queried_data():
    """Purchase dataset query results using query_id, supporting filters."""
    try:
        data = request.get_json()
        if not data or 'query_id' not in data:
            return jsonify({'error': 'Missing query_id'}), 400
        query_id = data['query_id']
        
        if not db.is_connected():
            return jsonify({'error': 'Database not connected'}), 503
            
        query_doc = db.queries_collection.find_one({"query_id": query_id})
        if not query_doc:
            return jsonify({'error': 'Query ID not found or invalid'}), 404
        if query_doc.get("status") == "purchased":
            logger.info(f"Query ID '{query_id}' already purchased.")
            return jsonify({
                "message": "This query has already been purchased.",
                "query_id": query_id,
                "data_response": query_doc.get("delivered_data_info", {})
            }), 200
            
        dataset_id = query_doc["dataset_id"]
        selected_attributes = query_doc["query_details_from_consumer"]["selected_attributes"]
        num_values_k = query_doc["query_details_from_consumer"]["num_values_per_attribute"]
        dp_epsilon = query_doc["dp_epsilon_for_price_and_privacy"]
        filters = query_doc["query_details_from_consumer"].get("filters", [])
        
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        # Retrieve filtered records using the new record-based approach
        filtered_rows = DatasetServiceSimple.get_dataset_records(
            dataset_id=dataset_id,
            selected_attributes=selected_attributes,
            filters=filters,
            # limit=num_values_k
        )
        
        # Apply simplified privacy to filtered rows
        combined_result, privacy_note = DatasetServiceSimple.apply_simple_privacy(
            dataset_id=dataset_id,
            selected_attributes=selected_attributes,
            num_values_k=num_values_k,
            epsilon=dp_epsilon,
            data_rows=filtered_rows,
            final_attribute_types=dataset['attribute_types']
        )
        # Extract privatized data from combined_result
        privatized_data = [row["privatized"] for row in combined_result]
        privatized_count = len(privatized_data)
        raw_data = [row["original"] for row in combined_result]
        raw_count = len([row["original"] for row in combined_result])
        data_response = {
            "privatized_data": privatized_data,
            "raw_data": raw_data,
            "privacy_note": privacy_note,
            "raw_count": raw_count,
            "privatized_count": privatized_count,
            "data_shape": {
                "rows": privatized_count,
                "columns": len(selected_attributes) if privatized_count else 0
            }
        }
        
        db.queries_collection.update_one(
            {"query_id": query_id},
            {
                "$set": {
                    "status": "purchased",
                    "delivered_data_info": data_response,
                    "purchase_timestamp": "2024-01-01T00:00:00Z"
                }
            }
        )
        
        logger.info(f"Data purchase successful for query ID '{query_id}'.")
        return jsonify({
            "message": "Purchase successful.",
            "query_id": query_id,
            "data_response": data_response
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error purchasing dataset: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/datasets/<string:dataset_id>/attribute-values', methods=['GET'])
def get_attribute_unique_values(dataset_id):
    """Return unique values for each attribute for dropdown filters."""
    try:
        dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
            
        all_attribute_names = dataset.get('all_attribute_names', [])
        attribute_types = dataset.get('attribute_types', {})
        unique_values = {}
        min_max = {}
        
        for attr in all_attribute_names:
            attr_type = attribute_types.get(attr, 'categorical')
            
            # For ALL attributes (both categorical and numeric), get ALL unique values
            pipeline = [
                {"$match": {"dataset_id": dataset_id}},
                {"$project": {f"data.{attr}": 1}},
                {"$group": {
                    "_id": None,
                    "values": {"$addToSet": f"$data.{attr}"}
                }}
            ]
            
            result = list(db.dataset_records_collection.aggregate(pipeline))
            if result:
                unique_values[attr] = sorted([v for v in result[0]['values'] if v not in ('', None)])
                
                # For numeric attributes, also calculate min/max
                if attr_type == 'numeric':
                    try:
                        nums = [float(v) for v in unique_values[attr] if v not in ('', None)]
                        if nums:
                            min_max[attr] = {'min': min(nums), 'max': max(nums)}
                    except Exception:
                        pass
            else:
                unique_values[attr] = []
        
        return jsonify({'unique_values': unique_values, 'min_max': min_max}), 200
    except Exception as e:
        logger.error(f"Error getting attribute unique values for {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/datasets/<string:dataset_id>/compare', methods=['POST'])
def compare_model_performance(dataset_id):
    """Train models on raw and noisy data and compare performance."""
    try:
        data = request.get_json()
        query_details = data.get('query_details', {})
        target_variable = data.get('target_variable')
        
        if not target_variable:
            return jsonify({'error': 'Target variable is required'}), 400

        dataset_metadata = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset_metadata:
            return jsonify({'error': 'Dataset not found'}), 404
        
        final_attribute_types = dataset_metadata.get('attribute_types', {})

        # Create a new list of attributes to fetch, ensuring the target variable is included.
        attributes_to_fetch = list(query_details.get('selected_attributes', [])) # Create a copy
        if target_variable not in attributes_to_fetch:
            attributes_to_fetch.append(target_variable)
            
        # Fetch the raw data using the corrected attribute list
        raw_data_rows = DatasetServiceSimple.get_dataset_records(
            dataset_id,
            selected_attributes=attributes_to_fetch, # Use the corrected list
            filters=query_details.get('filters', []),
            limit=query_details.get('num_values_per_attribute', 1000)
        )
        if not raw_data_rows:
            return jsonify({'error': 'No data found for the given query'}), 404

        # Generate the noisy (differentially private) data
        noisy_data_rows, _ = DatasetServiceSimple.apply_simple_privacy(
            dataset_id=dataset_id,
            selected_attributes=attributes_to_fetch, # Use the corrected list here as well
            num_values_k=query_details.get('num_values_per_attribute', 1000),
            epsilon=query_details.get('epsilon', 0.5),
            data_rows=raw_data_rows,
            final_attribute_types=final_attribute_types,
            target_variable=target_variable  # Pass the target variable
        )
        
        # Helper function to train and evaluate a model
        def train_and_evaluate(df, target):
            
            # Check for number of unique classes
            if df[target].nunique() < 2:
                raise ValueError(f"The selected data for the target '{target}' contains only one class. Please adjust your filters or query to include at least two classes.")

            # Ensure target column exists before dropping
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in the provided DataFrame for training.")

            X = df.drop(columns=[target])
            y = df[target]
            
            X = pd.get_dummies(X, drop_first=True, dummy_na=True) # handle potential new categoricals from noise
            
            return_X_cols = X.columns
            
            # Use stratify=y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            return metrics, return_X_cols

        # Convert to pandas DataFrame
        raw_df = pd.DataFrame(raw_data_rows)
        noisy_df = pd.DataFrame(noisy_data_rows)
        
        if target_variable not in raw_df.columns:
             return jsonify({'error': f'Target variable "{target_variable}" not found in dataset after fetching.'}), 400
        
        raw_metrics, raw_cols = train_and_evaluate(raw_df, target_variable)
        
        # Align columns of noisy_df to match raw_df's columns after one-hot encoding
        for col in raw_cols:
            if col not in noisy_df.columns:
                noisy_df[col] = 0
        
        # Make sure the target variable is also in the noisy dataframe
        if target_variable not in noisy_df.columns:
            # If target is missing from noisy data (unlikely but possible), add it from raw data
            if len(noisy_df) == len(raw_df):
                 noisy_df[target_variable] = raw_df[target_variable]
            else:
                # This case is tricky, for now, we'll raise an error
                raise ValueError("Target variable is missing in the noisy dataset and cannot be aligned.")

        # Ensure order of columns is the same, and include the target variable
        final_cols = list(raw_cols) + [target_variable]
        noisy_df = noisy_df[final_cols]

        noisy_metrics, _ = train_and_evaluate(noisy_df, target_variable)

        return jsonify({
            'raw_data_metrics': raw_metrics,
            'noisy_data_metrics': noisy_metrics
        }), 200

    except ValueError as e:
        logger.warning(f"Validation error during model comparison: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/datasets/<string:dataset_id>/count', methods=['POST'])
def count_filtered_records(dataset_id):
    """Count records that match the given filters."""
    try:
        data = request.get_json()
        filters = data.get('filters', [])
        
        # Validate dataset exists
        dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Count filtered records
        count = DatasetServiceSimple.count_dataset_records(dataset_id, filters)
        
        return jsonify({
            'dataset_id': dataset_id,
            'data_response': {
                'privatized_count': count,
                'raw_count': count,
                'filters_applied': len(filters),
                'total_count': dataset.get('n_total_records', 0)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error counting filtered records for dataset {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/marketplace/datasets/<string:dataset_id>/count-raw', methods=['POST'])
def count_raw_records(dataset_id):
    """Return the exact count of records matching the given filters (no noise)."""
    try:
        data = request.get_json()
        filters = data.get('filters', [])

        # Validate dataset exists
        dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        # Count filtered records
        count = DatasetServiceSimple.count_dataset_records(dataset_id, filters)

        return jsonify({
            'dataset_id': dataset_id,
            'filtered_count': count,
            'total_count': dataset.get('n_total_records', 0),
            'filters_applied': len(filters)
        }), 200
    except Exception as e:
        logger.error(f"Error counting raw records for dataset {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500