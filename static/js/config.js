/**
 * Configuration file for the Data Privacy Marketplace
 * Centralized configuration management
 */

const CONFIG = {
    // API Configuration
    API: {
        BASE_URL: 'http://localhost:5001',
        ENDPOINTS: {
            DATASETS: '/marketplace/datasets',
            DATASET_INFO: (id) => `/marketplace/datasets/${id}/info`,
            DATASET_VALUES: (id) => `/marketplace/datasets/${id}/attribute-values`,
            QUERY: (id) => `/marketplace/datasets/${id}/query`,
            PURCHASE: '/marketplace/purchase'
        },
        TIMEOUT: 30000, // 30 seconds
        RETRY_ATTEMPTS: 3
    },

    // UI Configuration
    UI: {
        ERROR_DISPLAY_DURATION: 5000, // 5 seconds
        LOADING_DELAY: 300, // 300ms minimum loading time
        MAX_RECORDS: 1000,
        MIN_RECORDS: 1,
        DEFAULT_RECORDS: 10,
        MIN_EPSILON: 0.1,
        MAX_EPSILON: 1.0,
        DEFAULT_EPSILON: 0.5,
        STEP_EPSILON: 0.1
    },

    // Validation Rules
    VALIDATION: {
        REQUIRED_ATTRIBUTES: 'Please select at least one attribute',
        INVALID_RECORDS: 'Please enter a valid number of records',
        INVALID_EPSILON: 'Please enter a valid privacy level (0.1 to 1.0)',
        NO_PRICE_QUOTE: 'Please get a price quote first'
    },

    // Messages
    MESSAGES: {
        LOADING_DATASETS: 'Loading datasets...',
        NO_DATASETS: 'No datasets available in the marketplace.',
        LOADING_ATTRIBUTES: 'Loading attributes...',
        PRICE_QUOTE_SUCCESS: 'Price quote generated successfully',
        PURCHASE_SUCCESS: 'Data purchased successfully',
        NO_DATA_AVAILABLE: 'No data available'
    },

    // CSS Classes
    CLASSES: {
        HIDDEN: 'hidden',
        SELECTED: 'selected',
        LOADING: 'loading',
        ERROR: 'error',
        SUCCESS: 'success'
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} 