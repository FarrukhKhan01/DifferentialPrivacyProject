from sklearn.model_selection import train_test_split
import uuid
import logging
import random
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import copy
import numpy as np  # for Laplace noise

from app.models.database import db
from app.utils.data_processing_simple import preprocess_data_and_get_counts_simple, mongo_to_dict
from app.utils.weight_calculation_simple import calculate_attribute_weights_simple
from app.utils.pricing import (
    calculate_prices_per_attribute,
    calculate_prices_per_attribute_value,
    calculate_price_for_consumer_query,
    calculate_final_price_from_dp_epsilon
)

logger = logging.getLogger(__name__)

class DatasetServiceSimple:
    """Simplified service class for dataset operations without pandas/numpy."""
    
    @staticmethod
    def create_dataset(
        dataset_name: str,
        csv_content: str,
        initial_p_max: float,
        initial_p_min: float,
        sensitive_attribute_name: str,
        attribute_types: Dict[str, str],
        numeric_attribute_ranges: Optional[Dict[str, List[float]]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a new dataset entry with individual records stored separately."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        dataset_id = str(uuid.uuid4())
        
        try:
            # Preprocess data and get counts
            (m_count, ni_counts, n_total_records, quasi_identifiers, 
             all_attribute_names, data_rows, final_attribute_types) = preprocess_data_and_get_counts_simple(
                dataset_name, csv_content, sensitive_attribute_name, attribute_types
            )
            
            # Calculate attribute weights using mutual information (original logic)
            weights_percentage, mi_scores = calculate_attribute_weights_simple(
                quasi_identifiers, all_attribute_names, sensitive_attribute_name, 
                dataset_id, data_rows, final_attribute_types
            )
            
            # Calculate prices per attribute using the original pricing logic
            p_max_attribute, p_min_attribute = calculate_prices_per_attribute(
                weights_percentage, initial_p_max, initial_p_min
            )
            
            # Calculate prices per attribute value
            p_max_attr_val_all, p_min_attr_val_all = calculate_prices_per_attribute_value(
                p_max_attribute, p_min_attribute, ni_counts
            )
            
            # Create dataset document (metadata only)
            dataset_doc = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "description": description,
                "sensitive_attribute_name": sensitive_attribute_name,
                "all_attribute_names": all_attribute_names,
                "quasi_identifiers": quasi_identifiers,
                "attribute_types": final_attribute_types,
                "numeric_attribute_ranges": numeric_attribute_ranges or {},
                "m_count": m_count,
                "ni_counts": ni_counts,
                "n_total_records": n_total_records,
                "weights_percentage": weights_percentage,
                "mi_scores": mi_scores,
                "prices_per_attribute": {
                    "max": p_max_attribute,
                    "min": p_min_attribute
                },
                "p_max_attr_val_all": p_max_attr_val_all,
                "p_min_attr_val_all": p_min_attr_val_all,
                "initial_p_max": initial_p_max,
                "initial_p_min": initial_p_min,
                "created_at": datetime.utcnow()
            }
            
            # Store dataset metadata
            db.datasets_collection.insert_one(dataset_doc)
            
            # Store individual records
            DatasetServiceSimple._store_dataset_records(dataset_id, data_rows, all_attribute_names)

            # --- Store DP records for a range of epsilon values ---
            epsilon_values = [round(x * 0.1, 1) for x in range(1, 11)]  # 0.1 to 1.0 inclusive
            for epsilon in epsilon_values:
                try:
                    privatized_records, _ = DatasetServiceSimple.apply_simple_privacy(
                        dataset_id=dataset_id,
                        selected_attributes=all_attribute_names,
                        num_values_k=len(data_rows),
                        epsilon=epsilon,
                        data_rows=data_rows,
                        final_attribute_types=final_attribute_types
                    )
                    dp_docs = []
                    for rec in privatized_records:
                        dp_docs.append({
                            "dataset_id": dataset_id,
                            "epsilon": epsilon,
                            "data": rec["privatized"]
                        })
                    if dp_docs:
                        db.dataset_records_dp.insert_many(dp_docs)
                    logger.info(f"Stored {len(dp_docs)} DP records for dataset {dataset_id} at epsilon={epsilon}")
                except Exception as e:
                    logger.error(f"Failed to generate DP records for epsilon={epsilon}: {e}")
            # --- End DP records generation ---
            
            logger.info(f"Dataset '{dataset_name}' created with ID: {dataset_id} and {n_total_records} records stored")
            
            return {
                "message": "Dataset processing completed.",
                "dataset_id": dataset_id,
                "mongo_id": str(dataset_doc.get("_id", "")),
                "details": {
                    "name": dataset_name,
                    "dataset_id": dataset_id,
                    "description": description,
                    "sensitive_attribute": sensitive_attribute_name,
                    "all_attribute_names": all_attribute_names,
                    "n_total_records": n_total_records,
                    "m_count": m_count,
                    "ni_counts": ni_counts,
                    "weights": weights_percentage,
                    "mi_scores": mi_scores,
                    "prices_per_attribute": {
                        "max": p_max_attribute,
                        "min": p_min_attribute
                    },
                    "p_max_initial": initial_p_max,
                    "p_min_initial": initial_p_min,
                    "status": "Completed"
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating dataset '{dataset_name}': {e}")
            raise
    
    @staticmethod
    def _store_dataset_records(dataset_id: str, data_rows: List[Dict], attribute_names: List[str]):
        """Store individual records in the dataset_records collection."""
        try:
            records_to_insert = []
            
            for index, row in enumerate(data_rows):
                record_doc = {
                    "dataset_id": dataset_id,
                    "record_id": f"{dataset_id}_record_{index}",
                    "record_index": index,
                    "data": row,
                    "created_at": datetime.utcnow()
                }
                records_to_insert.append(record_doc)
            
            # Insert records in batches for better performance
            batch_size = 1000
            for i in range(0, len(records_to_insert), batch_size):
                batch = records_to_insert[i:i + batch_size]
                db.dataset_records_collection.insert_many(batch)
            
            logger.info(f"Stored {len(records_to_insert)} records for dataset {dataset_id}")
            
        except Exception as e:
            logger.error(f"Error storing records for dataset {dataset_id}: {e}")
            raise
    
    @staticmethod
    def get_dataset_records(
        dataset_id: str,
        selected_attributes: List[str] = None,
        filters: List[Dict] = None,
        # limit: int = None,
        skip: int = 0
    ) -> List[Dict]:
        """Retrieve dataset records with optional filtering and attribute selection."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        try:
            # If no selected_attributes, use all columns except sensitive attribute
            if not selected_attributes:
                dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
                all_attrs = dataset.get("all_attribute_names", [])
                sensitive_attr = dataset.get("sensitive_attribute_name")
                selected_attributes = [a for a in all_attrs if a != sensitive_attr]
            
            # Build aggregation pipeline
            pipeline = [
                {"$match": {"dataset_id": dataset_id}}
            ]
            
            # Apply filters if provided
            if filters:
                match_conditions = {}
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
                
                if match_conditions:
                    pipeline.append({"$match": match_conditions})
            
            # Project only selected attributes if specified
            if selected_attributes:
                projection = {
                    "dataset_id": 1,
                    "record_id": 1,
                    "record_index": 1
                }
                for attr in selected_attributes:
                    projection[f"data.{attr}"] = 1
                pipeline.append({"$project": projection})
            
            # Add pagination
            if skip > 0:
                pipeline.append({"$skip": skip})
            # if limit:
            #     pipeline.append({"$limit": limit})
            
            # Execute aggregation
            records = list(db.dataset_records_collection.aggregate(pipeline))
            
            # Transform to expected format
            transformed_records = []
            for record in records:
                if selected_attributes:
                    # Only include selected attributes
                    filtered_data = {}
                    for attr in selected_attributes:
                        if f"data.{attr}" in record:
                            filtered_data[attr] = record[f"data.{attr}"]
                        elif "data" in record and attr in record["data"]:
                            filtered_data[attr] = record["data"][attr]
                        else:
                            filtered_data[attr] = ""
                    transformed_records.append({'data': filtered_data})
                else:
                    # Include all data
                    transformed_records.append({'data': record.get("data", {})})
            
            return transformed_records
            
        except Exception as e:
            logger.error(f"Error retrieving records for dataset {dataset_id}: {e}")
            raise
    
    @staticmethod
    def count_dataset_records(dataset_id: str, filters: List[Dict] = None) -> int:
        """Count records in a dataset with optional filtering."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        try:
            pipeline = [
                {"$match": {"dataset_id": dataset_id}}
            ]
            
            # Apply filters if provided
            if filters:
                match_conditions = {}
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
                
                if match_conditions:
                    pipeline.append({"$match": match_conditions})
            
            pipeline.append({"$count": "total"})
            
            result = list(db.dataset_records_collection.aggregate(pipeline))
            return result[0]["total"] if result else 0
            
        except Exception as e:
            logger.error(f"Error counting records for dataset {dataset_id}: {e}")
            raise
    
    @staticmethod
    def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
        if dataset:
            return mongo_to_dict(dataset)
        return None
    
    @staticmethod
    def list_datasets() -> List[Dict[str, Any]]:
        """List all available datasets sorted by creation time (latest first)."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        try:
            # Get all datasets sorted by created_at in descending order (latest first)
            datasets = list(db.datasets_collection.find({}).sort("created_at", -1))
            
            # Convert to the format expected by the frontend
            formatted_datasets = []
            for dataset in datasets:
                try:
                    # Handle both old (name) and new (dataset_name) field names
                    dataset_name = dataset.get("dataset_name") or dataset.get("name") or "Unknown Dataset"
                    
                    formatted_dataset = {
                        "dataset_id": dataset.get("dataset_id", "unknown"),
                        "dataset_name": dataset_name,  # Keep original field name
                        "name": dataset_name,  # Frontend expects 'name'
                        "description": dataset.get("description", ""),
                        "n_total_records": dataset.get("n_total_records", 0),
                        "total_records": dataset.get("n_total_records", 0),  # Alternative field name
                        "total_attributes": len(dataset.get("all_attribute_names", [])),
                        "all_attribute_names": dataset.get("all_attribute_names", []),
                        "sensitive_attribute_name": dataset.get("sensitive_attribute_name"),
                        "sensitive_attribute": dataset.get("sensitive_attribute_name"),  # Alternative field name
                        "created_at": dataset.get("created_at", "2024-01-01T00:00:00Z")
                    }
                    formatted_datasets.append(formatted_dataset)
                except Exception as e:
                    logger.warning(f"Error formatting dataset {dataset.get('dataset_id', 'unknown')}: {e}")
                    continue
            
            return formatted_datasets
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            raise Exception(f"Failed to list datasets: {e}")
    
    @staticmethod
    def calculate_query_price(
        dataset_id: str,
        selected_attributes: List[str],
        num_values_k: int,
        dp_epsilon: float,
        filters: List[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate price for a dataset query based on filtered data."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        dataset = db.datasets_collection.find_one({"dataset_id": dataset_id})
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # If the list is empty, default to using all non-sensitive attributes.
        if not selected_attributes:
            logger.info("No attributes selected by consumer. Defaulting to all non-sensitive attributes.")
            all_attrs = dataset.get("all_attribute_names", [])
            sensitive_attr = dataset.get("sensitive_attribute_name")
            selected_attributes = [attr for attr in all_attrs if attr != sensitive_attr]

        # Validate selected attributes
        available_attrs = dataset.get("all_attribute_names", [])
        invalid_attrs = [attr for attr in selected_attributes if attr not in available_attrs]
        if invalid_attrs:
            raise ValueError(f"Invalid attributes: {invalid_attrs}. Available: {available_attrs}")
        
        # Count filtered records to determine pricing basis
        filtered_count = DatasetServiceSimple.count_dataset_records(dataset_id, filters)
        total_count = dataset.get('n_total_records', 0)
        
        # Calculate scarcity factor - if filtered data is much smaller than total, price should be higher
        scarcity_factor = 1.0
        if total_count > 0 and filtered_count > 0:
            # If filtered data is less than 50% of total, apply scarcity pricing
            if filtered_count < total_count * 0.5:
                scarcity_factor = min(2.0, total_count / filtered_count)  # Cap at 2x price
                logger.info(f"Applying scarcity factor {scarcity_factor} for filtered data ({filtered_count}/{total_count})")
        
        # Cap the requested number of records to what's actually available
        actual_k = min(num_values_k, filtered_count)
        if actual_k != num_values_k:
            logger.warning(f"Requested {num_values_k} records but only {filtered_count} available. Using {actual_k}.")
        
        # Calculate price for each attribute based on filtered data
        num_values_per_attribute_k = {attr: actual_k for attr in selected_attributes}
        
        # Use original pricing calculation but apply scarcity factor
        p_min_k_query, p_max_k_query = calculate_price_for_consumer_query(
            selected_attributes,
            num_values_per_attribute_k,
            dataset["p_max_attr_val_all"],
            dataset["p_min_attr_val_all"],
            dataset["ni_counts"]
        )
        
        # Apply scarcity factor to prices
        p_min_k_query *= scarcity_factor
        p_max_k_query *= scarcity_factor
        
        # Calculate final price based on DP epsilon
        final_price = calculate_final_price_from_dp_epsilon(p_min_k_query, p_max_k_query, dp_epsilon)
        
        return {
            "dataset_id": dataset_id,
            "selected_attributes": selected_attributes,
            "num_values_k": actual_k,
            "dp_epsilon": dp_epsilon,
            "p_min_k_query": p_min_k_query,
            "p_max_k_query": p_max_k_query,
            "final_price": final_price,
            "filtered_count": filtered_count,
            "total_count": total_count,
            "scarcity_factor": scarcity_factor,
            "price_breakdown": {
                "base_price_range": [p_min_k_query / scarcity_factor, p_max_k_query / scarcity_factor],
                "scarcity_factor": scarcity_factor,
                "adjusted_price_range": [p_min_k_query, p_max_k_query],
                "privacy_adjustment": dp_epsilon,
                "final_price": final_price
            }
        }
    
    @staticmethod
    def purchase_dataset(
        dataset_id: str,
        selected_attributes: List[str],
        num_values_k: int,
        dp_epsilon: float,
        payment_amount: float
    ) -> Dict[str, Any]:
        """Purchase dataset query results."""
        # Calculate expected price
        price_info = DatasetServiceSimple.calculate_query_price(
            dataset_id, selected_attributes, num_values_k, dp_epsilon
        )
        
        expected_price = price_info["final_price"]
        
        # Validate payment
        if abs(payment_amount - expected_price) > 0.01:
            raise ValueError(f"Payment amount {payment_amount} does not match expected price {expected_price}")
        
        # Store query record
        query_id = str(uuid.uuid4())
        query_doc = {
            "query_id": query_id,
            "dataset_id": dataset_id,
            "selected_attributes": selected_attributes,
            "num_values_k": num_values_k,
            "dp_epsilon": dp_epsilon,
            "payment_amount": payment_amount,
            "timestamp": "2024-01-01T00:00:00Z"  # Simplified timestamp
        }
        
        db.queries_collection.insert_one(query_doc)
        
        logger.info(f"Query {query_id} purchased for dataset {dataset_id}")
        
        return {
            "query_id": query_id,
            "dataset_id": dataset_id,
            "payment_confirmed": True,
            "payment_amount": payment_amount,
            "message": "Purchase successful. Data processing initiated."
        }
    

    @staticmethod
    def calculate_numeric_avg(
        dataset_id: str,
        attribute_name: str,
        epsilon: float,
        filters: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Calculate differentially private average for a numeric attribute.
        
        Args:
            dataset_id: ID of the dataset
            attribute_name: Name of the numeric attribute
            epsilon: Privacy budget
            filters: Optional filters to apply to records
            
        Returns:
            Dictionary with:
                - true_avg: True average (before noise)
                - noisy_avg: Differentially private average
                - epsilon: Privacy parameter used
                - message: Privacy explanation
        """
        if not db.is_connected():
            raise Exception("Database connection not available")
        
        dataset = DatasetServiceSimple.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Validate attribute
        attribute_types = dataset.get("attribute_types", {})
        if attribute_name not in attribute_types or attribute_types[attribute_name] != 'numeric':
            raise ValueError(f"Attribute '{attribute_name}' is not numeric")
        
        # Get min/max from dataset metadata
        numeric_ranges = dataset.get("numeric_attribute_ranges", {})
        if attribute_name not in numeric_ranges:
            raise ValueError(f"Min/max range for attribute '{attribute_name}' not available")
        
        attr_min, attr_max = numeric_ranges[attribute_name]
        sensitivity = attr_max - attr_min
        
        # Count filtered records
        filtered_count = DatasetServiceSimple.count_dataset_records(dataset_id, filters)
        if filtered_count == 0:
            return {
                "true_avg": 0,
                "noisy_avg": 0,
                "epsilon": epsilon,
                "message": "No records match the filter criteria"
            }
        
        # Calculate true sum and average
        pipeline = [
            {"$match": {"dataset_id": dataset_id}}
        ]
        
        # Apply filters if provided
        if filters:
            match_conditions = {}
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
            
            if match_conditions:
                pipeline.append({"$match": match_conditions})
        
        # Add grouping to calculate sum
        pipeline.append({
            "$group": {
                "_id": None,
                "total_sum": {"$sum": f"$data.{attribute_name}"}
            }
        })
        
        result = list(db.dataset_records_collection.aggregate(pipeline))
        total_sum = result[0]["total_sum"] if result else 0
        true_avg = total_sum / filtered_count
        
        # Apply Laplace noise for differential privacy
        # Sensitivity = (max - min) since changing one record can change sum by at most this amount
        scale = sensitivity / epsilon
        noise = random.expovariate(1/scale) - random.expovariate(1/scale)  # Laplace noise
        noisy_avg = true_avg + (noise / filtered_count)
        
        return {
            "true_avg": true_avg,
            "noisy_avg": noisy_avg,
            "epsilon": epsilon,
            "message": (
                f"Differentially private average calculated with ε={epsilon:.3f}. "
                "True average: {true_avg:.2f}, Noisy average: {noisy_avg:.2f}. "
                "Noise added to protect individual privacy."
            )
        }

    @staticmethod
    @staticmethod
    def apply_simple_privacy(
        dataset_id: str,
        selected_attributes: List[str],
        num_values_k: int,
        epsilon: float,
        data_rows: List[Dict],
        final_attribute_types: Dict[str, str],
        target_variable: str = None
    ) -> Tuple[List[Dict], str]:
        """Apply Laplace differential privacy to numeric attributes only, and return original + privatized data."""
        
        data_to_process = copy.deepcopy(data_rows)
        logger.info(f"Applying Laplace DP to dataset {dataset_id} with epsilon={epsilon}")

        # Validation
        if epsilon < 0.1 or epsilon > 1:
            raise ValueError(f"Epsilon must be between 0.1 and 1, got {epsilon}")
        if num_values_k <= 0:
            raise ValueError(f"Number of values k must be positive, got {num_values_k}")
        if not selected_attributes:
            raise ValueError("No attributes selected for query")

        # Filter to selected attributes
        filtered_data = []
        for row in data_to_process:
            row_data = row.get('data', row)  # unwrap if needed
            filtered_row = {attr: row_data.get(attr, '') for attr in selected_attributes}
            filtered_data.append(filtered_row)

        # Sampling
        if len(filtered_data) > num_values_k:
            if target_variable and target_variable in selected_attributes:
                df = pd.DataFrame(filtered_data)
                if df[target_variable].nunique() > 1:
                    sampled_df, _ = train_test_split(
                        df, train_size=num_values_k, stratify=df[target_variable], random_state=42
                    )
                    filtered_data = sampled_df.to_dict('records')
                else:
                    filtered_data = random.sample(filtered_data, num_values_k)
            else:
                filtered_data = random.sample(filtered_data, num_values_k)

        # Apply Laplace noise to numeric attributes only
        noisy_data = []
        sensitivity = 1  # assuming global sensitivity = 1 for simplicity

        for row in filtered_data:
            noisy_row = {}
            for attr in selected_attributes:
                original_value = row.get(attr, '')
                if final_attribute_types.get(attr) == 'numeric' and attr != target_variable:
                    try:
                        value = float(original_value)
                        scale = sensitivity / epsilon
                        noise = np.random.laplace(loc=0, scale=scale)
                        privatized_value = value + noise
                        privatized_value = max(0, privatized_value)  # Ensure no negative values
                        noisy_row[attr] = str(round(privatized_value, 4))
                    except (ValueError, TypeError):
                        noisy_row[attr] = original_value  # fallback
                else:
                    noisy_row[attr] = original_value  # leave categorical untouched
            noisy_data.append(noisy_row)

        # Construct side-by-side response (original vs privatized)
        combined_result = []
        for original, privatized in zip(filtered_data, noisy_data):
            combined_result.append({
                "original": original,
                "privatized": privatized
            })

        # Build privacy note
        privacy_note = (
            f"Differential privacy applied with Laplace mechanism (ε={epsilon:.3f}) to numeric attributes. "
            f"Categorical attributes remain unchanged. Returned values are approximate."
        )

        return combined_result, privacy_note

    @staticmethod
    def count_dp_records(dataset_id: str, epsilon: float, filters: List[Dict] = None) -> int:
        """Count DP records in dataset_records_dp with optional filtering and epsilon."""
        if not db.is_connected():
            raise Exception("Database connection not available")
        try:
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
            count = db.dataset_records_dp.count_documents(match_conditions)
            return count
        except Exception as e:
            logger.error(f"Error counting DP records for dataset {dataset_id}, epsilon {epsilon}: {e}")
            raise