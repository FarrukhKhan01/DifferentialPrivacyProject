/**
 * Reports Service - Handles API calls for reporting functionality
 */

class ReportsService {
    constructor() {
        this.baseUrl = '/reports';
    }

    /**
     * Fetch dashboard report data
     */
    async getDashboardReport() {
        try {
            const response = await fetch(`${this.baseUrl}/dashboard`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching dashboard report:', error);
            throw error;
        }
    }

    /**
     * Fetch dataset-specific report
     * @param {string} datasetId - The dataset ID
     */
    async getDatasetReport(datasetId) {
        try {
            const response = await fetch(`${this.baseUrl}/dataset/${datasetId}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching dataset report:', error);
            throw error;
        }
    }

    /**
     * Fetch revenue report data
     * @param {string} startDate - Start date (YYYY-MM-DD)
     * @param {string} endDate - End date (YYYY-MM-DD)
     */
    async getRevenueReport(startDate = null, endDate = null) {
        try {
            const params = new URLSearchParams();
            if (startDate) params.append('start_date', startDate);
            if (endDate) params.append('end_date', endDate);
            
            const url = `${this.baseUrl}/revenue${params.toString() ? '?' + params.toString() : ''}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching revenue report:', error);
            throw error;
        }
    }

    /**
     * Fetch privacy report data
     */
    async getPrivacyReport() {
        try {
            const response = await fetch(`${this.baseUrl}/privacy`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching privacy report:', error);
            throw error;
        }
    }

    /**
     * Fetch usage report data
     * @param {string} startDate - Start date (YYYY-MM-DD)
     * @param {string} endDate - End date (YYYY-MM-DD)
     */
    async getUsageReport(startDate = null, endDate = null) {
        try {
            const params = new URLSearchParams();
            if (startDate) params.append('start_date', startDate);
            if (endDate) params.append('end_date', endDate);
            
            const url = `${this.baseUrl}/usage${params.toString() ? '?' + params.toString() : ''}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching usage report:', error);
            throw error;
        }
    }

    /**
     * Export report data as CSV
     * @param {string} reportType - Type of report to export
     * @param {Object} data - Report data to export
     */
    exportToCSV(reportType, data) {
        try {
            let csvContent = '';
            
            switch (reportType) {
                case 'revenue':
                    csvContent = this.convertRevenueToCSV(data);
                    break;
                case 'privacy':
                    csvContent = this.convertPrivacyToCSV(data);
                    break;
                case 'usage':
                    csvContent = this.convertUsageToCSV(data);
                    break;
                default:
                    throw new Error(`Unknown report type: ${reportType}`);
            }
            
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', `${reportType}_report_${new Date().toISOString().split('T')[0]}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            console.error('Error exporting to CSV:', error);
            throw error;
        }
    }

    /**
     * Convert revenue data to CSV format
     */
    convertRevenueToCSV(data) {
        let csv = 'Dataset,Revenue,Queries,Avg Price\n';
        
        if (data.revenue_by_dataset) {
            data.revenue_by_dataset.forEach(dataset => {
                csv += `${dataset.name},${dataset.revenue},${dataset.queries},${dataset.avg_price}\n`;
            });
        }
        
        return csv;
    }

    /**
     * Convert privacy data to CSV format
     */
    convertPrivacyToCSV(data) {
        let csv = 'Dataset,Privacy Score,Quasi Identifiers,Sensitive Attribute\n';
        
        if (data.privacy_by_dataset) {
            data.privacy_by_dataset.forEach(dataset => {
                csv += `${dataset.name},${dataset.privacy_score},${dataset.quasi_identifiers},${dataset.sensitive_attribute}\n`;
            });
        }
        
        return csv;
    }

    /**
     * Convert usage data to CSV format
     */
    convertUsageToCSV(data) {
        let csv = 'Attribute,Queries,Percentage\n';
        
        if (data.popular_attributes) {
            data.popular_attributes.forEach(attr => {
                csv += `${attr.attribute},${attr.queries},${attr.percentage}\n`;
            });
        }
        
        return csv;
    }

    /**
     * Format currency values
     * @param {number} value - Value to format
     * @param {string} currency - Currency code (default: USD)
     */
    formatCurrency(value, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(value);
    }

    /**
     * Format percentage values
     * @param {number} value - Value to format
     * @param {number} decimals - Number of decimal places
     */
    formatPercentage(value, decimals = 1) {
        return `${value.toFixed(decimals)}%`;
    }

    /**
     * Format large numbers with K, M, B suffixes
     * @param {number} value - Value to format
     */
    formatNumber(value) {
        if (value >= 1000000000) {
            return (value / 1000000000).toFixed(1) + 'B';
        } else if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        } else if (value >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        }
        return value.toString();
    }

    /**
     * Get color for different privacy levels
     * @param {string} level - Privacy level
     */
    getPrivacyColor(level) {
        const colors = {
            'high': '#22c55e',
            'medium': '#fbbf24',
            'low': '#ef4444'
        };
        return colors[level] || '#6b7280';
    }

    /**
     * Get color for different revenue levels
     * @param {number} value - Revenue value
     * @param {number} maxValue - Maximum value for comparison
     */
    getRevenueColor(value, maxValue) {
        const percentage = (value / maxValue) * 100;
        if (percentage >= 80) return '#22c55e';
        if (percentage >= 50) return '#fbbf24';
        return '#ef4444';
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReportsService;
} else {
    // Make available globally
    window.ReportsService = ReportsService;
} 