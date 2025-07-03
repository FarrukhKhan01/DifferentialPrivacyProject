import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# def calculate_final_price_from_dp_epsilon(p_min_k_query: float, p_max_k_query: float, dp_epsilon: float) -> float:
#     """
#     Calculate final price based on differential privacy epsilon value.
    
#     Args:
#         p_min_k_query: Minimum price for k query
#         p_max_k_query: Maximum price for k query
#         dp_epsilon: Differential privacy epsilon parameter
        
#     Returns:
#         Final calculated price
#     """
#     if dp_epsilon <= 0 or dp_epsilon > 1:
#         raise ValueError(f"DP epsilon must be between 0 and 1, got {dp_epsilon}")
    
#     if p_min_k_query >= p_max_k_query:
#         logger.warning(f"P_min ({p_min_k_query}) >= P_max ({p_max_k_query}). Using P_min as final price.")
#         return p_min_k_query
    
#     # Price decreases as epsilon increases (more privacy = higher price)
#     # Formula: final_price = p_min + (p_max - p_min) * (1 - epsilon)
#     final_price = p_min_k_query + (p_max_k_query - p_min_k_query) * (1 - dp_epsilon)
    
#     logger.debug(f"Price calculation: P_min={p_min_k_query}, P_max={p_max_k_query}, ε={dp_epsilon}, Final={final_price}")
    
#     return final_price


def calculate_final_price_from_dp_epsilon(p_min_k_query: float, p_max_k_query: float, dp_epsilon: float) -> float:
    """
    Calculate final price based on differential privacy epsilon value.
    
    Args:
        p_min_k_query: Minimum price for k query
        p_max_k_query: Maximum price for k query
        dp_epsilon: Differential privacy epsilon parameter
        
    Returns:
        Final calculated price
    """
    if dp_epsilon < 0.1 or dp_epsilon > 1:
        raise ValueError(f"DP epsilon must be between 0.1 and 1, got {dp_epsilon}")
    
    if p_min_k_query >= p_max_k_query:
        logger.warning(f"P_min ({p_min_k_query}) >= P_max ({p_max_k_query}). Using P_min as final price.")
        return p_min_k_query
    
    # Formula from the document: final_price = p_min + (p_max - p_min) * epsilon
    final_price = p_min_k_query + (p_max_k_query - p_min_k_query) * dp_epsilon
    
    logger.debug(f"Price calculation: P_min={p_min_k_query}, P_max={p_max_k_query}, ε={dp_epsilon}, Final={final_price}")
    
    return final_price



def calculate_prices_per_attribute(weights_percentage: Dict[str, float], p_max_initial: float, p_min_initial: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate min and max prices for each attribute based on weights.
    
    Args:
        weights_percentage: Dictionary mapping attribute names to weight percentages
        p_max_initial: Initial maximum price
        p_min_initial: Initial minimum price
        
    Returns:
        Tuple of (p_max_attribute, p_min_attribute) dictionaries
    """
    p_max_attribute = {}
    p_min_attribute = {}
    
    for attr, weight_percent in weights_percentage.items():
        p_max_attribute[attr] = (weight_percent * p_max_initial) / 100.0
        p_min_attribute[attr] = (weight_percent * p_min_initial) / 100.0
    
    return p_max_attribute, p_min_attribute

def calculate_prices_per_attribute_value(p_max_attr: Dict[str, float], p_min_attr: Dict[str, float], ni_counts: Dict[str, int]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate prices per attribute value based on number of unique values.
    
    Args:
        p_max_attr: Dictionary mapping attributes to maximum prices
        p_min_attr: Dictionary mapping attributes to minimum prices
        ni_counts: Dictionary mapping attribute names to number of unique values
        
    Returns:
        Tuple of (p_max_attr_value, p_min_attr_value) dictionaries
    """
    p_max_attr_value = {}
    p_min_attr_value = {}
    
    for attr_name in p_max_attr.keys():
        num_values = ni_counts.get(attr_name, 1)
        if num_values == 0:
            num_values = 1
        
        p_max_attr_value[attr_name] = p_max_attr[attr_name] / num_values
        p_min_attr_value[attr_name] = p_min_attr[attr_name] / num_values
    
    return p_max_attr_value, p_min_attr_value

# def calculate_price_for_consumer_query(
#     selected_attributes: List[str],
#     num_values_per_attribute_k: Dict[str, int],
#     p_max_attr_val_all: Dict[str, float],
#     p_min_attr_val_all: Dict[str, float]
# ) -> Tuple[float, float]:
#     """
#     Calculate total price for a consumer query.
    
#     Args:
#         selected_attributes: List of selected attributes
#         num_values_per_attribute_k: Dictionary mapping attributes to number of values requested
#         p_max_attr_val_all: Dictionary mapping attributes to maximum price per value
#         p_min_attr_val_all: Dictionary mapping attributes to minimum price per value
        
#     Returns:
#         Tuple of (total_p_min, total_p_max)
#     """
#     new_p_max_k_query = 0.0
#     new_p_min_k_query = 0.0
    
#     for attr_name in selected_attributes:
#         if attr_name in p_max_attr_val_all and attr_name in p_min_attr_val_all:
#             k = num_values_per_attribute_k.get(attr_name, 1)
#             new_p_max_k_query += k * p_max_attr_val_all[attr_name]
#             new_p_min_k_query += k * p_min_attr_val_all[attr_name]
#         else:
#             logger.warning(f"Selected attribute '{attr_name}' not found in pre-calculated attribute value prices or has zero price. Price component will be 0.")
    
#     logger.debug(f"Price calculation for attributes {selected_attributes}: P_min={new_p_min_k_query}, P_max={new_p_max_k_query}")
    
#     return new_p_min_k_query, new_p_max_k_query 

def calculate_price_for_consumer_query(
    selected_attributes: List[str],
    num_values_per_attribute_k: Dict[str, int],
    p_max_attr_val_all: Dict[str, float],
    p_min_attr_val_all: Dict[str, float],
    ni_counts: Dict[str, int]  # <<< The missing 5th parameter is now added
) -> Tuple[float, float]:
    """
    Calculate total price for a consumer query, capping the number of values
    to the maximum available for each attribute.
    """
    new_p_max_k_query = 0.0
    new_p_min_k_query = 0.0
    
    for attr_name in selected_attributes:
        if attr_name in p_max_attr_val_all and attr_name in p_min_attr_val_all:
            # Get the number of values the consumer requested for this attribute
            k = num_values_per_attribute_k.get(attr_name, 1)

            # Get the total number of unique values that actually exist for this attribute
            total_available_values = ni_counts.get(attr_name, 1)

            # --- FIX: Cap the requested values (k) at the maximum available ---
            if k > total_available_values:
                logger.warning(f"Consumer requested {k} values for '{attr_name}', but only {total_available_values} are available. Capping at max.")
                k = total_available_values
            
            # Calculate the price using the (potentially capped) value of k
            new_p_max_k_query += k * p_max_attr_val_all[attr_name]
            new_p_min_k_query += k * p_min_attr_val_all[attr_name]
        else:
            logger.warning(f"Selected attribute '{attr_name}' not found in pre-calculated attribute value prices or has zero price. Price component will be 0.")
    
    logger.debug(f"Price calculation for attributes {selected_attributes}: P_min={new_p_min_k_query}, P_max={new_p_max_k_query}")
    
    return new_p_min_k_query, new_p_max_k_query
