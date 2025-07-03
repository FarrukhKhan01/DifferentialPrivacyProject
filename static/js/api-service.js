/**
 * API Service for Data Privacy Marketplace
 * Handles all API calls and data communication
 */

class ApiService {
    constructor() {
        this.baseUrl = CONFIG.API.BASE_URL;
        this.timeout = CONFIG.API.TIMEOUT;
        this.retryAttempts = CONFIG.API.RETRY_ATTEMPTS;
    }

    /**
     * Make an API request with error handling and retry logic
     * @param {string} url - The URL to request
     * @param {Object} options - Fetch options
     * @returns {Promise} Promise with the response data
     */
    async makeRequest(url, options = {}) {
        const fullUrl = url.startsWith('http') ? url : `${this.baseUrl}${url}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            timeout: this.timeout
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            const response = await Utils.retry(
                () => this.fetchWithTimeout(fullUrl, requestOptions),
                this.retryAttempts
            );

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new ApiError(
                    errorData.error || `HTTP ${response.status}: ${response.statusText}`,
                    response.status,
                    errorData
                );
            }

            return await response.json();
        } catch (error) {
            if (error instanceof ApiError) {
                throw error;
            }
            throw new ApiError(`Network error: ${error.message}`, 0, { originalError: error });
        }
    }

    /**
     * Fetch with timeout
     * @param {string} url - The URL to fetch
     * @param {Object} options - Fetch options
     * @returns {Promise} Promise with the response
     */
    async fetchWithTimeout(url, options) {
        const { timeout, ...fetchOptions } = options;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
            const response = await fetch(url, {
                ...fetchOptions,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }
            throw error;
        }
    }

    /**
     * Get all available datasets
     * @returns {Promise<Array>} Promise with datasets array
     */
    async getDatasets() {
        return this.makeRequest(CONFIG.API.ENDPOINTS.DATASETS);
    }

    /**
     * Get dataset information
     * @param {string} datasetId - The dataset ID
     * @returns {Promise<Object>} Promise with dataset info
     */
    async getDatasetInfo(datasetId) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.DATASET_INFO(datasetId));
    }

    /**
     * Get dataset attribute values
     * @param {string} datasetId - The dataset ID
     * @returns {Promise<Object>} Promise with attribute values
     */
    async getDatasetAttributeValues(datasetId) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.DATASET_VALUES(datasetId));
    }

    /**
     * Submit a query for price quote
     * @param {string} datasetId - The dataset ID
     * @param {Object} queryData - Query parameters
     * @returns {Promise<Object>} Promise with price quote
     */
    async submitQuery(datasetId, queryData) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.QUERY(datasetId), {
            method: 'POST',
            body: JSON.stringify(queryData)
        });
    }

    /**
     * Purchase data
     * @param {Object} purchaseData - Purchase parameters
     * @returns {Promise<Object>} Promise with purchase result
     */
    async purchaseData(purchaseData) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.PURCHASE, {
            method: 'POST',
            body: JSON.stringify(purchaseData)
        });
    }

    /**
     * Validate query parameters before submission
     * @param {Object} queryData - Query data to validate
     * @returns {Object} Validation result
     */
    validateQueryData(queryData) {
        const errors = [];

        // Validate selected attributes
        if (!queryData.selected_attributes || queryData.selected_attributes.length === 0) {
            errors.push(CONFIG.VALIDATION.REQUIRED_ATTRIBUTES);
        }

        // Validate number of values
        const numValues = parseInt(queryData.num_values_per_attribute);
        if (!numValues || numValues < CONFIG.UI.MIN_RECORDS || numValues > CONFIG.UI.MAX_RECORDS) {
            errors.push(CONFIG.VALIDATION.INVALID_RECORDS);
        }

        // Validate epsilon
        const epsilon = parseFloat(queryData.epsilon);
        if (!epsilon || epsilon < CONFIG.UI.MIN_EPSILON || epsilon > CONFIG.UI.MAX_EPSILON) {
            errors.push(CONFIG.VALIDATION.INVALID_EPSILON);
        }

        // Validate filters
        if (queryData.filters) {
            const invalidFilters = queryData.filters.filter(filter => 
                !filter.attr || filter.value === '' || filter.value === undefined
            );
            if (invalidFilters.length > 0) {
                errors.push('Some filters have invalid values');
            }
        }

        return {
            isValid: errors.length === 0,
            errors: errors
        };
    }

    /**
     * Prepare query data for submission
     * @param {Object} rawData - Raw query data
     * @returns {Object} Prepared query data
     */
    prepareQueryData(rawData) {
        return {
            selected_attributes: rawData.selected_attributes || [],
            num_values_per_attribute: parseInt(rawData.num_values_per_attribute) || CONFIG.UI.DEFAULT_RECORDS,
            epsilon: parseFloat(rawData.epsilon) || CONFIG.UI.DEFAULT_EPSILON,
            filters: (rawData.filters || []).filter(filter => 
                filter.attr && filter.value !== '' && filter.value !== undefined
            )
        };
    }
}

/**
 * Custom API Error class
 */
class ApiError extends Error {
    constructor(message, status = 0, data = {}) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
        this.timestamp = new Date();
    }

    /**
     * Get a user-friendly error message
     * @returns {string} User-friendly error message
     */
    getUserMessage() {
        if (this.status === 0) {
            return 'Network error. Please check your connection and try again.';
        }
        
        if (this.status >= 500) {
            return 'Server error. Please try again later.';
        }
        
        if (this.status === 404) {
            return 'Resource not found.';
        }
        
        if (this.status === 403) {
            return 'Access denied.';
        }
        
        if (this.status === 401) {
            return 'Authentication required.';
        }
        
        return this.message || 'An unexpected error occurred.';
    }

    /**
     * Check if the error is retryable
     * @returns {boolean} True if error can be retried
     */
    isRetryable() {
        // Retry on network errors and server errors
        return this.status === 0 || this.status >= 500;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ApiService, ApiError };
} 