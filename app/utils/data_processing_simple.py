import csv
import io
import logging
import re
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

def mongo_to_dict(doc):
    """Convert MongoDB document to dictionary with proper type handling."""
    if doc and '_id' in doc:
        doc['_id'] = str(doc['_id'])
    if doc: 
        for key, value in doc.items():
            if isinstance(value, dict): 
                mongo_to_dict(value)
            elif isinstance(value, list): 
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        mongo_to_dict(item)
    return doc

def detect_csv_format(csv_content_string: str) -> Tuple[str, str]:
    """
    Detect CSV delimiter and quote character from content.
    
    Returns:
        Tuple of (delimiter, quotechar)
    """
    # Get first few lines for analysis
    lines = csv_content_string.split('\n')[:5]
    if not lines:
        return ',', '"'
    
    # Count different delimiters
    delimiter_counts = {
        ',': 0,
        ';': 0,
        '\t': 0,
        '|': 0
    }
    
    for line in lines:
        if line.strip():
            delimiter_counts[','] += line.count(',')
            delimiter_counts[';'] += line.count(';')
            delimiter_counts['\t'] += line.count('\t')
            delimiter_counts['|'] += line.count('|')
    
    # Find most common delimiter
    delimiter = max(delimiter_counts, key=delimiter_counts.get)
    
    # Detect quote character
    quote_counts = {'"': 0, "'": 0}
    for line in lines:
        if line.strip():
            quote_counts['"'] += line.count('"')
            quote_counts["'"] += line.count("'")
    
    quotechar = max(quote_counts, key=quote_counts.get) if max(quote_counts.values()) > 0 else '"'
    
    logger.info(f"Detected CSV format: delimiter='{delimiter}', quotechar='{quotechar}'")
    return delimiter, quotechar

def clean_csv_content(csv_content_string: str) -> str:
    """
    Clean CSV content by removing BOM and normalizing line endings.
    """
    # Remove BOM if present
    if csv_content_string.startswith('\ufeff'):
        csv_content_string = csv_content_string[1:]
    
    # Normalize line endings
    csv_content_string = csv_content_string.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove trailing empty lines
    csv_content_string = csv_content_string.rstrip('\n')
    
    return csv_content_string

def preprocess_data_and_get_counts_simple(
    dataset_name: str, 
    csv_content_string: str, 
    sensitive_attribute_name: str, 
    attribute_types_from_request: Dict[str, str]
) -> Tuple[int, Dict[str, int], int, List[str], List[str], List[Dict], Dict[str, str]]:
    """
    Enhanced CSV processing with better format detection and error handling.
    
    Args:
        dataset_name: Name of the dataset
        csv_content_string: CSV content as string
        sensitive_attribute_name: Name of the sensitive attribute
        attribute_types_from_request: Dictionary mapping attribute names to types
        
    Returns:
        Tuple containing: m_count, ni_counts, n_total_records, quasi_identifiers, 
                         all_attribute_names, data_rows, final_attribute_types
    """
    logger.debug(f"Preprocessing dataset: {dataset_name}")
    
    try:
        # Clean the CSV content
        csv_content_string = clean_csv_content(csv_content_string)
        
        if not csv_content_string.strip():
            raise ValueError("CSV content is empty or contains only whitespace.")
        
        # Detect CSV format
        delimiter, quotechar = detect_csv_format(csv_content_string)
        
        # Try to parse with detected format
        try:
            csv_reader = csv.reader(io.StringIO(csv_content_string), 
                                  delimiter=delimiter, 
                                  quotechar=quotechar,
                                  skipinitialspace=True)
            rows = list(csv_reader)
        except Exception as e:
            # Fallback to comma delimiter
            logger.warning(f"Failed to parse with detected format, trying comma delimiter: {e}")
            csv_reader = csv.reader(io.StringIO(csv_content_string), 
                                  delimiter=',', 
                                  quotechar='"',
                                  skipinitialspace=True)
            rows = list(csv_reader)
        
        if not rows:
            raise ValueError("CSV data empty after loading.")
        
        # Validate header row
        first_row = rows[0]
        if not first_row or not any(cell.strip() for cell in first_row):
            raise ValueError("CSV must have a valid header row with non-empty column names.")
        
        # Clean header names (remove quotes and extra whitespace)
        headers = [cell.strip().strip('"\'') for cell in first_row]
        
        # Check for duplicate headers
        if len(headers) != len(set(headers)):
            duplicates = [h for h in set(headers) if headers.count(h) > 1]
            raise ValueError(f"Duplicate column names found: {duplicates}")
        
        # Validate sensitive attribute
        if sensitive_attribute_name not in headers:
            available_columns = ', '.join(headers)
            raise ValueError(f"Sensitive attribute '{sensitive_attribute_name}' not found in columns. Available columns: {available_columns}")
        
        logger.info(f"Header validated for '{dataset_name}'. Columns: {len(headers)}")
        
        # Process data rows
        data_rows_raw = rows[1:]
        
        # Filter out empty rows and validate row lengths
        data_rows = []
        for i, row in enumerate(data_rows_raw, start=2):  # Start at 2 for line numbers
            if not any(cell.strip() for cell in row):
                continue  # Skip empty rows
                
            if len(row) != len(headers):
                logger.warning(f"Row {i} has {len(row)} columns, expected {len(headers)}. Padding or truncating.")
                # Pad or truncate row to match header length
                if len(row) < len(headers):
                    row.extend([''] * (len(headers) - len(row)))
                else:
                    row = row[:len(headers)]
            
            # Create row dictionary
            row_dict = {}
            for j, header in enumerate(headers):
                value = row[j].strip() if j < len(row) else ''
                row_dict[header] = value
            
            data_rows.append(row_dict)
        
        if not data_rows:
            raise ValueError("No valid data rows found after processing.")
        
        all_attribute_names = headers
        
        # Validate attribute types
        for attr_key in attribute_types_from_request.keys():
            if attr_key not in all_attribute_names:
                raise ValueError(f"Attribute '{attr_key}' in 'attribute_types' not found in columns: {all_attribute_names}")

        # Calculate statistics
        m_count = len(all_attribute_names)
        ni_counts = {}
        final_attribute_types = {} 

        for col in all_attribute_names:
            # Determine attribute type
            if col in attribute_types_from_request:
                final_attribute_types[col] = attribute_types_from_request[col]
            else: 
                # Enhanced type inference
                sample_values = [row[col] for row in data_rows[:100] if col in row and row[col].strip()]
                if not sample_values:
                    final_attribute_types[col] = 'categorical'
                else:
                    numeric_count = 0
                    for val in sample_values:
                        try:
                            float(val)
                            numeric_count += 1
                        except (ValueError, AttributeError):
                            pass
                    
                    # If more than 70% of values are numeric, consider it numeric
                    if numeric_count / len(sample_values) > 0.7:
                        final_attribute_types[col] = 'numeric'
                    else:
                        final_attribute_types[col] = 'categorical'
                
                logger.info(f"Inferred type for '{col}' as '{final_attribute_types[col]}' for {dataset_name}.")
            
            # Count unique values
            unique_values = set()
            for row in data_rows:
                if col in row and row[col].strip():
                    unique_values.add(row[col].strip())
            ni_counts[col] = len(unique_values)

        n_total_records = len(data_rows)
        quasi_identifiers = [attr for attr in all_attribute_names if attr != sensitive_attribute_name]
        
        logger.info(f"Preprocessing for '{dataset_name}' completed successfully. Records: {n_total_records}, Attributes: {len(all_attribute_names)}")
        
        return m_count, ni_counts, n_total_records, quasi_identifiers, all_attribute_names, data_rows, final_attribute_types
        
    except Exception as e:
        raise ValueError(f"Error processing CSV for '{dataset_name}': {str(e)}") 