import logging
import math
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

def calculate_mutual_information_simple(attribute_name: str, sensitive_attribute_name: str, data_rows: List[Dict], final_attribute_types: Dict[str, str]) -> float:
    """
    Calculate mutual information between an attribute and sensitive attribute using only Python standard libraries.
    
    Args:
        attribute_name: Name of the attribute to calculate MI for
        sensitive_attribute_name: Name of the sensitive attribute
        data_rows: List of dictionaries representing data rows
        final_attribute_types: Dictionary mapping attribute names to types
        
    Returns:
        Mutual information score
    """
    try:
        # Extract values for both attributes
        attr_values = []
        sensitive_values = []
        
        for row in data_rows:
            if attribute_name in row and sensitive_attribute_name in row:
                attr_val = str(row[attribute_name]).strip()
                sensitive_val = str(row[sensitive_attribute_name]).strip()
                
                if attr_val and sensitive_val:  # Skip empty values
                    attr_values.append(attr_val)
                    sensitive_values.append(sensitive_val)
        
        if not attr_values or not sensitive_values:
            return 0.0001
        
        # Calculate joint and marginal distributions
        joint_counts = Counter(zip(attr_values, sensitive_values))
        attr_counts = Counter(attr_values)
        sensitive_counts = Counter(sensitive_values)
        
        total_samples = len(attr_values)
        
        if total_samples == 0:
            return 0.0001
        
        # Calculate mutual information: MI(X,Y) = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
        mi_score = 0.0
        
        for (attr_val, sensitive_val), joint_count in joint_counts.items():
            if joint_count == 0:
                continue
                
            # Calculate probabilities
            p_xy = joint_count / total_samples
            p_x = attr_counts[attr_val] / total_samples
            p_y = sensitive_counts[sensitive_val] / total_samples
            
            # Avoid log(0) by checking if probabilities are valid
            if p_x > 0 and p_y > 0 and p_xy > 0:
                mi_score += p_xy * math.log(p_xy / (p_x * p_y))
        
        return abs(mi_score) if not math.isnan(mi_score) else 0.0001
        
    except Exception as e:
        logger.error(f"Error calculating MI for '{attribute_name}' vs '{sensitive_attribute_name}': {e}")
        return 0.0001

def calculate_attribute_weights_simple(
    quasi_identifiers: List[str], 
    all_attribute_names: List[str], 
    sensitive_attribute_name: str, 
    dataset_id: str, 
    data_rows: List[Dict], 
    final_attribute_types: Dict[str, str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate attribute weights using mutual information, preserving the original logic.
    
    Args:
        quasi_identifiers: List of quasi-identifier attributes
        all_attribute_names: List of all attribute names
        sensitive_attribute_name: Name of the sensitive attribute
        dataset_id: Dataset identifier for logging
        data_rows: List of dictionaries representing data rows
        final_attribute_types: Dictionary mapping attribute names to types
        
    Returns:
        Tuple of (percentage_weights, mi_scores)
    """
    logger.debug(f"Calculating weights for {dataset_id}. SA: '{sensitive_attribute_name}'. QIDs: {quasi_identifiers}")
    
    if sensitive_attribute_name not in final_attribute_types:
        logger.error(f"SA '{sensitive_attribute_name}' type unknown for {dataset_id}.")
        num_attrs = len(all_attribute_names)
        default_w = 100 / num_attrs if num_attrs > 0 else 0
        return {attr: default_w for attr in all_attribute_names}, {attr: 0.0 for attr in all_attribute_names}

    # Calculate mutual information for each quasi-identifier vs sensitive attribute
    mi_scores_qids_vs_sa = {}
    for qid_attr in quasi_identifiers:
        if qid_attr == sensitive_attribute_name:
            continue
        mi_scores_qids_vs_sa[qid_attr] = calculate_mutual_information_simple(
            qid_attr, sensitive_attribute_name, data_rows, final_attribute_types
        )

    # Initialize all attributes with small default MI score
    all_mi_scores_for_normalization = {attr: 0.00001 for attr in all_attribute_names}
    all_mi_scores_for_normalization.update(mi_scores_qids_vs_sa)

    # Assign sensitive attribute weight based on average MI of other attributes (QIDs)
    if quasi_identifiers and mi_scores_qids_vs_sa:
        valid_qid_mi_scores = [s for s in mi_scores_qids_vs_sa.values() if s is not None and s > 0.00001]
        if valid_qid_mi_scores:
            # Calculate mean of valid QID MI scores
            sa_mi_score = sum(valid_qid_mi_scores) / len(valid_qid_mi_scores)
            all_mi_scores_for_normalization[sensitive_attribute_name] = sa_mi_score
        else:
            logger.warning(f"No valid QID MI scores to calculate mean for SA weight in {dataset_id}.")
    elif sensitive_attribute_name in all_mi_scores_for_normalization:
        logger.info(f"No QIDs with MI scores, SA '{sensitive_attribute_name}' keeps default MI for {dataset_id}.")
    
    # Calculate total MI for normalization
    total_mi_all = sum(s for s in all_mi_scores_for_normalization.values() if s is not None and s > 0.00001)
    
    # Normalize weights
    normalized_weights = {}
    if total_mi_all > 0:
        for attr_name in all_attribute_names:
            normalized_weights[attr_name] = (all_mi_scores_for_normalization.get(attr_name, 0) / total_mi_all)
    else:
        logger.warning(f"Total MI for normalization is zero for {dataset_id}. Using equal weights.")
        num_all_attrs = len(all_attribute_names)
        default_weight = 1.0 / num_all_attrs if num_all_attrs > 0 else 0
        for attr_name in all_attribute_names:
            normalized_weights[attr_name] = default_weight
            
    # Convert to percentage weights
    percentage_weights = {attr: weight * 100 for attr, weight in normalized_weights.items()}
    
    logger.debug(f"Final MI scores for {dataset_id}: {all_mi_scores_for_normalization}")
    logger.debug(f"Percentage weights for {dataset_id}: {percentage_weights}")
    
    return percentage_weights, all_mi_scores_for_normalization 