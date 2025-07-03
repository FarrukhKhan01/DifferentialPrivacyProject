from flask import Blueprint, request, jsonify
from app.models.database import db
from app.services.dataset_service_simple import DatasetServiceSimple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

analytics_bp = Blueprint('analytics', __name__)
logger = logging.getLogger(__name__)

def apply_filters(records, filters):
    # Simple filter application: supports ==, >, >=, <, <=
    def match(record, filter_):
        attr = filter_['attribute']
        op = filter_['op']
        value = filter_['value']
        record_value = record.get(attr)
        if record_value is None:
            return False
        try:
            if op == '==':
                return record_value == value
            elif op == '>':
                return float(record_value) > float(value)
            elif op == '>=':
                return float(record_value) >= float(value)
            elif op == '<':
                return float(record_value) < float(value)
            elif op == '<=':
                return float(record_value) <= float(value)
            else:
                return False
        except Exception:
            return False
    for f in filters:
        records = [r for r in records if match(r, f)]
    return records

def add_laplace_noise(value, epsilon):
    # Add Laplace noise for differential privacy
    if epsilon <= 0:
        return value
    scale = 1.0 / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

def add_noise_to_dataframe(df, epsilon, numeric_columns):
    """Add Laplace noise to numeric columns in a dataframe."""
    noisy_df = df.copy()
    
    for col in numeric_columns:
        if col in noisy_df.columns:
            # Add noise to each value in the column
            noise = np.random.laplace(0, 1/epsilon, len(noisy_df))
            noisy_df[col] = noisy_df[col] + noise
    
    return noisy_df

def determine_model_type(target_column, df):
    """Determine if the target variable is classification or regression."""
    unique_values = df[target_column].nunique()
    total_values = len(df[target_column])
    
    # If less than 20% unique values or less than 10 unique values, treat as classification
    if unique_values / total_values < 0.2 or unique_values < 10:
        return 'classification'
    else:
        return 'regression'

def prepare_data_for_ml(df, target_column, feature_columns):
    """Prepare data for machine learning by handling categorical variables."""
    df_ml = df.copy()
    
    # Handle categorical variables
    label_encoders = {}
    for col in feature_columns:
        if df_ml[col].dtype == 'object':
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
    
    # Handle target variable
    if df_ml[target_column].dtype == 'object':
        le_target = LabelEncoder()
        df_ml[target_column] = le_target.fit_transform(df_ml[target_column].astype(str))
        label_encoders[target_column] = le_target
    
    # Convert to numeric
    for col in feature_columns + [target_column]:
        df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
    
    # Remove rows with NaN values
    df_ml = df_ml.dropna(subset=feature_columns + [target_column])
    
    return df_ml, label_encoders

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='classification'):
    """Train and evaluate a model, returning performance metrics."""
    try:
        if model_type == 'classification':
            # Try multiple classification models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
        else:
            # Try multiple regression models
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
        
        best_model = None
        best_score = -1
        best_model_name = None
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                if model_type == 'classification':
                    score = model.score(X_test, y_test)  # Accuracy for classification
                else:
                    score = model.score(X_test, y_test)  # R² for regression
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue
        
        if best_model is None:
            return None, None, None
        
        # Get predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        if model_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return best_model, metrics, best_model_name
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate_model: {e}")
        return None, None, None

@analytics_bp.route('/analytics/compare-aggregate', methods=['POST'])
def compare_aggregate():
    """
    Compare aggregate (sum/count) for multiple groups, optionally with differential privacy.
    """
    try:
        data = request.get_json()
        dataset_id = data['dataset_id']
        aggregation = data['aggregation']
        attribute = data['attribute']
        comparisons = data['comparisons']
        compare_raw_vs_private = data.get('compare_raw_vs_private', False)
        epsilon = float(data.get('epsilon', 0.5))

        # Fetch all records for the dataset
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        all_attributes = dataset.get('all_attribute_names', [])
        if attribute not in all_attributes:
            return jsonify({'error': f'Attribute {attribute} not found in dataset'}), 400

        # Get all records (limit to 10000 for performance)
        records = DatasetServiceSimple.get_dataset_records(
            dataset_id=dataset_id,
            selected_attributes=all_attributes,
            filters=[],
            # limit=10000
        )

        results = []
        for group in comparisons:
            label = group.get('label', 'Group')
            filters = group.get('filters', [])
            group_records = apply_filters(records, filters) if filters else records

            if aggregation == 'count':
                # Always get raw count from dataset_records
                raw_value = DatasetServiceSimple.count_dataset_records(dataset_id, filters)
                result = {'label': label, 'raw': raw_value}
                if compare_raw_vs_private:
                    epsilon_values = [round(x * 0.1, 1) for x in range(1, 11)]
                    dp_counts = {}
                    for eps in epsilon_values:
                        dp_counts[str(eps)] = DatasetServiceSimple.count_dp_records(dataset_id, eps, filters)
                    result['dp_counts'] = dp_counts
            elif aggregation == 'sum':
                try:
                    raw_value = sum(float(r.get(attribute, 0)) for r in group_records if r.get(attribute) is not None)
                    result = {'label': label, 'raw': raw_value}
                except Exception:
                    raw_value = None
                    result = {'label': label, 'raw': raw_value}
            else:
                return jsonify({'error': 'Invalid aggregation type'}), 400

            if compare_raw_vs_private and aggregation == 'sum' and raw_value is not None:
                private_value = add_laplace_noise(raw_value, epsilon)
                result['private'] = private_value
            results.append(result)

        return jsonify({'results': results}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analytics_bp.route('/analytics/model-comparison', methods=['POST'])
def model_comparison():
    """
    Compare model performance when trained on original data and tested on noisy data.
    This demonstrates the impact of differential privacy on model utility.
    """
    try:
        data = request.get_json()
        dataset_id = data['dataset_id']
        target_column = data['target_column']
        feature_columns = data.get('feature_columns', [])
        epsilon_values = data.get('epsilon_values', [0.1, 0.5, 1.0])
        test_size = data.get('test_size', 0.2)
        max_samples = data.get('max_samples', 50000)  # Increased limit for larger datasets
        
        # Fetch dataset
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Get all attributes if feature_columns not specified
        all_attributes = dataset.get('all_attribute_names', [])
        if not feature_columns:
            feature_columns = [attr for attr in all_attributes if attr != target_column]
        
        # Validate columns
        if target_column not in all_attributes:
            return jsonify({'error': f'Target column {target_column} not found in dataset'}), 400
        
        for col in feature_columns:
            if col not in all_attributes:
                return jsonify({'error': f'Feature column {col} not found in dataset'}), 400
        
        # Get records
        records = DatasetServiceSimple.get_dataset_records(
            dataset_id=dataset_id,
            selected_attributes=all_attributes,
            filters=[],
            # limit=max_samples
        )
        
        if not records:
            return jsonify({'error': 'No records found for the dataset'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([record['data'] for record in records])
        
        # Prepare data for ML
        df_ml, label_encoders = prepare_data_for_ml(df, target_column, feature_columns)
        
        if len(df_ml) < 100:
            return jsonify({'error': 'Insufficient data for meaningful model comparison'}), 400
        
        # Determine model type
        model_type = determine_model_type(target_column, df_ml)
        
        # Split data
        X = df_ml[feature_columns]
        y = df_ml[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model on original (clean) data
        original_model, _, original_model_name = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, model_type
        )
        
        if original_model is None:
            return jsonify({'error': 'Failed to train model on original data'}), 500
        
        # Evaluate on original test data (baseline)
        y_pred_orig = original_model.predict(X_test)
        
        # Calculate baseline metrics
        if model_type == 'classification':
            original_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_orig),
                'precision': precision_score(y_test, y_pred_orig, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_orig, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred_orig, average='weighted', zero_division=0)
            }
            original_confusion = confusion_matrix(y_test, y_pred_orig).tolist()
        else:
            original_metrics = {
                'r2_score': r2_score(y_test, y_pred_orig),
                'mse': mean_squared_error(y_test, y_pred_orig),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_orig))
            }
            original_confusion = None
        
        # Get class labels for classification
        class_labels = None
        if model_type == 'classification':
            if target_column in label_encoders:
                class_labels = list(label_encoders[target_column].classes_)
            else:
                class_labels = sorted(list(set(y_test)))
        
        # Results storage
        comparison_results = {
            'original': {
                'model_name': original_model_name,
                'metrics': original_metrics,
                'data_size': len(df_ml),
                'confusion_matrix': original_confusion,
                'class_labels': class_labels
            },
            'noisy_comparisons': []
        }
        
        # Test the same model on noisy test data with different epsilon values
        for epsilon in epsilon_values:
            try:
                # Add noise to test data (not training data)
                X_test_noisy = add_noise_to_dataframe(X_test, epsilon, feature_columns)
                
                # Use the same model trained on clean data to predict on noisy test data
                y_pred_noisy = original_model.predict(X_test_noisy)
                
                # Calculate metrics on noisy test data
                if model_type == 'classification':
                    noisy_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred_noisy),
                        'precision': precision_score(y_test, y_pred_noisy, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred_noisy, average='weighted', zero_division=0),
                        'f1_score': f1_score(y_test, y_pred_noisy, average='weighted', zero_division=0)
                    }
                    noisy_confusion = confusion_matrix(y_test, y_pred_noisy).tolist()
                else:
                    noisy_metrics = {
                        'r2_score': r2_score(y_test, y_pred_noisy),
                        'mse': mean_squared_error(y_test, y_pred_noisy),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_noisy))
                    }
                    noisy_confusion = None
                
                comparison_results['noisy_comparisons'].append({
                    'epsilon': epsilon,
                    'model_name': original_model_name,  # Same model as original
                    'metrics': noisy_metrics,
                    'performance_degradation': calculate_performance_degradation(
                        original_metrics, noisy_metrics, model_type
                    ),
                    'confusion_matrix': noisy_confusion
                })
                
            except Exception as e:
                logger.warning(f"Failed to process epsilon {epsilon}: {e}")
                continue
        
        # Add summary statistics
        comparison_results['summary'] = {
            'model_type': model_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'total_comparisons': len(comparison_results['noisy_comparisons']),
            'average_degradation': calculate_average_degradation(comparison_results['noisy_comparisons'])
        }
        
        return jsonify(comparison_results), 200
        
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        return jsonify({'error': str(e)}), 500

def calculate_performance_degradation(original_metrics, noisy_metrics, model_type):
    """Calculate performance degradation between original and noisy models."""
    degradation = {}
    
    if model_type == 'classification':
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in original_metrics and metric in noisy_metrics:
                original_val = original_metrics[metric]
                noisy_val = noisy_metrics[metric]
                if original_val > 0:
                    degradation[metric] = ((original_val - noisy_val) / original_val) * 100
                else:
                    degradation[metric] = 0
    else:
        for metric in ['r2_score']:
            if metric in original_metrics and metric in noisy_metrics:
                original_val = original_metrics[metric]
                noisy_val = noisy_metrics[metric]
                if original_val > 0:
                    degradation[metric] = ((original_val - noisy_val) / original_val) * 100
                else:
                    degradation[metric] = 0
        
        # For MSE and RMSE, higher is worse, so we calculate differently
        for metric in ['mse', 'rmse']:
            if metric in original_metrics and metric in noisy_metrics:
                original_val = original_metrics[metric]
                noisy_val = noisy_metrics[metric]
                if original_val > 0:
                    degradation[metric] = ((noisy_val - original_val) / original_val) * 100
                else:
                    degradation[metric] = 0
    
    return degradation

def calculate_average_degradation(noisy_comparisons):
    """Calculate average performance degradation across all comparisons."""
    if not noisy_comparisons:
        return {}
    
    all_degradations = {}
    
    for comparison in noisy_comparisons:
        degradation = comparison.get('performance_degradation', {})
        for metric, value in degradation.items():
            if metric not in all_degradations:
                all_degradations[metric] = []
            all_degradations[metric].append(value)
    
    average_degradation = {}
    for metric, values in all_degradations.items():
        average_degradation[metric] = sum(values) / len(values)
    
    return average_degradation
