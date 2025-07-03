class ConsumerInterface {
    constructor() {
        this.API_BASE_URL = 'http://localhost:5001';
        this.selectedDatasetId = null;
        this.selectedAttributes = [];
        this.currentPriceQuote = null;
        this.attributeTypes = {};
        this.filters = [];
        this.attributeUniqueValues = {};
        this.attributeMinMax = {};
        this.queryPreviewTimer = null;
        this.lastQueryPreview = null;
        this.dataTableRows = [];
        this.dataTablePage = 1;
        this.dataTablePageSize = 20;
        this.dataTableSortColumn = null;
        this.dataTableSortDirection = 'asc';
        this.dataTableFilterText = '';

        this.init();
    }

    /**
     * Initialize the interface
     */
    init() {
        this.bindEventListeners();
        this.loadDatasets();
    }

    /**
     * Bind all event listeners
     * * *** FIX 1: Added safety checks to prevent the script from crashing
     * *** if an element is not found on page load.
     */
    bindEventListeners() {
        const getPriceBtn = document.getElementById('getPriceBtn');
        if (getPriceBtn) {
            getPriceBtn.addEventListener('click', () => this.getPriceQuote());
        }

        const purchaseBtn = document.getElementById('purchaseBtn');
        if (purchaseBtn) {
            purchaseBtn.addEventListener('click', () => this.purchaseData());
        }

        const addFilterBtn = document.getElementById('addFilterBtn');
        if (addFilterBtn) {
            addFilterBtn.addEventListener('click', () => this.addFilter());
        }

        const numValuesInput = document.getElementById('numValues');
        if (numValuesInput) {
            numValuesInput.addEventListener('input', () => this.debouncedQueryPreview());
        }

        const epsilonInput = document.getElementById('epsilon');
        if (epsilonInput) {
            epsilonInput.addEventListener('input', () => this.debouncedQueryPreview());
        }

        document.addEventListener('filterChanged', () => this.debouncedQueryPreview());
    }


    /**
     * Select a dataset and load its attributes
     * * *** FIX 2: Integrated the logic to show and populate the
     * *** new Model Comparison section directly into this function.
     */
    async selectDataset(dataset) {
        this.selectedDatasetId = dataset.dataset_id;

        // Store globally for query builder
        window.selectedDatasetId = dataset.dataset_id;

        // Update UI to show selection
        this.updateDatasetSelection(dataset);

        // Show attribute selection
        this.showElement('attributeSelection');
        document.getElementById('selectedDatasetInfo').textContent = `Dataset: ${dataset.name || 'N/A'}`;

        // Load attributes and unique values
        await this.loadAttributesAndValues(dataset.dataset_id);

        // Show query configuration
        this.showElement('queryConfiguration');

        // Hide other sections until they are needed
        this.hideElement('priceQuote');
        this.hideElement('results');

        // Initialize available records count
        this.updateQueryPreview();

        // Emit event for Smart Query Builder
        const event = new CustomEvent('datasetSelected', {
            detail: {
                dataset: dataset,
                attributes: this.attributeTypes
            }
        });
        window.dispatchEvent(event);
    }


    // ... (The rest of your functions: updateDatasetSelection, loadAttributesAndValues, etc. are perfectly fine and do not need changes)
    // Just ensure the corrected bindEventListeners() and selectDataset() functions from above are in your class.
    // I am including all the functions here for completeness so you can copy-paste the entire class.

    /**
     * Update available records count independently
     */
    async updateAvailableRecordsCount() {
        if (!this.selectedDatasetId) {
            return;
        }

        try {
            const countData = await this.countFilteredRecords();
            this.updateAvailableRecordsDisplay(countData);
        } catch (error) {
            console.warn('Failed to update available records count:', error);
        }
    }

    /**
     * Debounced query preview to avoid excessive API calls
     */
    debouncedQueryPreview() {
        if (this.queryPreviewTimer) {
            clearTimeout(this.queryPreviewTimer);
        }
        this.queryPreviewTimer = setTimeout(() => {
            this.updateQueryPreview();
        }, 500);
    }

    /**
     * Debounced available records count update
     */
    debouncedAvailableRecordsUpdate() {
        if (this.availableRecordsTimer) {
            clearTimeout(this.availableRecordsTimer);
        }
        this.availableRecordsTimer = setTimeout(() => {
            this.updateAvailableRecordsCount();
        }, 300);
    }

    /**
     * Count filtered records to show available data
     */
    async countFilteredRecords() {
        if (!this.selectedDatasetId) {
            return null;
        }

        try {
            const filters = this.filters.filter(f => f.attr && f.value !== '' && f.value !== undefined);
            const response = await fetch(`${this.API_BASE_URL}/marketplace/datasets/${this.selectedDatasetId}/count`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filters: filters })
            });

            if (response.ok) {
                const data = await response.json();
                return data;
            } else {
                console.warn('Failed to count filtered records');
                return null;
            }
        } catch (error) {
            console.warn('Error counting filtered records:', error);
            return null;
        }
    }

    /**
     * Update query preview with real-time information
     */
    async updateQueryPreview() {
        if (!this.selectedDatasetId || this.selectedAttributes.length === 0) {
            return;
        }

        try {
            const queryData = this.buildQueryData();
            const queryKey = JSON.stringify(queryData);

            if (this.lastQueryPreview === queryKey) {
                return;
            }
            this.lastQueryPreview = queryKey;
            this.showQueryPreviewLoading();

            // Get filtered record count first
            const countData = await this.countFilteredRecords();

            const response = await fetch(`${this.API_BASE_URL}/marketplace/datasets/${this.selectedDatasetId}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(queryData)
            });

            if (response.ok) {
                const quote = await response.json();
                this.displayQueryPreview(quote, countData);
            } else {
                this.hideQueryPreview();
            }
        } catch (error) {
            console.warn('Query preview failed:', error);
            this.hideQueryPreview();
        }
    }

    /**
     * Update available records display
     */
    updateAvailableRecordsDisplay(countData) {
        const availableCountElement = document.getElementById('availableCount');
        const availableInfoElement = document.getElementById('availableRecordsInfo');
        const numValuesInput = document.getElementById('numValues');

        if (!availableCountElement || !availableInfoElement || !numValuesInput) {
            return;
        }

        if (countData && countData.data_response) {
            const availableCount = countData.data_response.privatized_count;
            const totalCount = countData.data_response.total_count;
            const requestedCount = parseInt(numValuesInput.value) || 10;

            availableCountElement.textContent = `${availableCount} (${totalCount} total)`;

            // Update styling based on comparison
            availableInfoElement.className = 'available-records-info';

            if (requestedCount > availableCount) {
                availableInfoElement.classList.add('warning');
                availableCountElement.innerHTML = `<span class="warning-text">${availableCount} (${totalCount} total) - Warning: You're requesting more than available!</span>`;
            } else if (availableCount === 0) {
                availableInfoElement.classList.add('error');
                availableCountElement.innerHTML = `<span class="error-text">0 - No records match your filters!</span>`;
            }

            // Update input max value
            numValuesInput.max = availableCount;

            // If requested is more than available, adjust it
            if (requestedCount > availableCount && availableCount > 0) {
                numValuesInput.value = availableCount;
                this.debouncedQueryPreview();
            }
        } else {
            availableCountElement.textContent = 'Calculating...';
            availableInfoElement.className = 'available-records-info';
        }
    }

    /**
     * Display real-time query preview
     */
    displayQueryPreview(quote, countData = null) {
        const previewContainer = document.getElementById('queryPreview');
        if (!previewContainer) {
            const queryConfig = document.getElementById('queryConfiguration');
            const previewDiv = document.createElement('div');
            previewDiv.id = 'queryPreview';
            previewDiv.className = 'query-preview bg-blue-50 border border-blue-200 rounded-md p-4 mt-4';
            queryConfig.appendChild(previewDiv);
        }

        // Update available records display
        this.updateAvailableRecordsDisplay(countData);

        const container = document.getElementById('queryPreview');
        const requestedRecords = parseInt(document.getElementById('numValues').value) || 10;
        const availableRecords = countData && countData.data_response ? countData.data_response.privatized_count : 'Calculating...';
        const totalRecords = countData && countData.data_response ? countData.data_response.total_count : 'Unknown';

        let recordsInfo = `🔢 Records: ${requestedRecords}`;
        if (countData) {
            recordsInfo = `🔢 Records: ${requestedRecords} of ${availableRecords} available (${totalRecords} total)`;
            if (requestedRecords > availableRecords) {
                recordsInfo += ` ⚠️ Requested more than available`;
            }
        }

        // Add pricing information
        let pricingInfo = `💰 Estimated Price: $${quote.final_price?.toFixed(2) || 'Calculating...'}`;
        if (quote.scarcity_factor && quote.scarcity_factor > 1.0) {
            pricingInfo += ` ⚡ Scarcity: ${quote.scarcity_factor.toFixed(1)}x`;
        }
        if (quote.price_breakdown) {
            const basePrice = quote.price_breakdown.base_price_range?.[0] || 0;
            const adjustedPrice = quote.price_breakdown.adjusted_price_range?.[0] || 0;
            if (adjustedPrice > basePrice) {
                pricingInfo += ` 📈 Base: $${basePrice.toFixed(2)}`;
            }
        }

        container.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <h4 class="text-sm font-medium text-blue-800 mb-2">🔍 Query Preview</h4>
                    <div class="text-xs text-blue-700 space-y-1">
                        <div>📊 Selected: ${this.selectedAttributes.length} attributes</div>
                        <div>${recordsInfo}</div>
                        <div>🛡️ Privacy: ε = ${document.getElementById('epsilon').value}</div>
                        <div>${pricingInfo}</div>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-xs text-blue-600">Real-time</div>
                    <div class="text-xs text-blue-500">Updated just now</div>
                </div>
            </div>
        `;
        container.style.display = 'block';
    }

    /**
     * Show loading state for query preview
     */
    showQueryPreviewLoading() {
        const container = document.getElementById('queryPreview');
        if (container) {
            container.innerHTML = `
                <div class="flex items-center text-blue-700">
                    <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                    <span class="text-sm">Updating preview...</span>
                </div>
            `;
            container.style.display = 'block';
        }
    }

    /**
     * Hide query preview
     */
    hideQueryPreview() {
        const container = document.getElementById('queryPreview');
        if (container) {
            container.style.display = 'none';
        }
    }

    /**
     * Build query data for API calls
     */
    buildQueryData() {
        return {
            selected_attributes: this.selectedAttributes,
            num_values_per_attribute: parseInt(document.getElementById('numValues').value) || 10,
            epsilon: parseFloat(document.getElementById('epsilon').value) || 0.5,
            filters: this.filters.filter(f => f.attr && f.value !== '' && f.value !== undefined)
        };
    }

    /**
     * Load available datasets from the API
     */
    async loadDatasets() {
        this.showLoading('loadingDatasets');
        this.hideElement('noDatasets');
        this.hideElement('datasetList');

        try {
            const response = await fetch(`${this.API_BASE_URL}/marketplace/datasets`);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to load datasets');
            }
            const datasets = Array.isArray(data) ? data : (data.datasets || []);
            this.displayDatasets(datasets);
        } catch (error) {
            this.showError(`Failed to load datasets: ${error.message}`);
        } finally {
            this.hideLoading('loadingDatasets');
        }
    }

    /**
     * Display datasets in the UI
     */
    displayDatasets(datasets) {
        const container = document.getElementById('datasetList');
        if (!datasets || datasets.length === 0) {
            this.showElement('noDatasets');
            return;
        }
        container.innerHTML = '';
        datasets.forEach(dataset => {
            const card = this.createDatasetCard(dataset);
            container.appendChild(card);
        });
        this.showElement('datasetList');
    }

    /**
     * Create a dataset card element
     */
    createDatasetCard(dataset) {
        const card = document.createElement('div');
        card.className = 'dataset-card';
        card.onclick = () => this.selectDataset(dataset);
        const createdDate = dataset.created_at ? new Date(dataset.created_at).toLocaleDateString() : 'Unknown';
        card.innerHTML = `
            <div class="dataset-info">
                <div class="dataset-details">
                    <h3 class="dataset-name">${dataset.dataset_name || dataset.name}</h3>
                    <p class="dataset-description">${dataset.description || 'No description available'}</p>
                    <div class="dataset-meta">
                        <span>📊 ${dataset.n_total_records || 'Unknown'} records</span>
                        <span>📅 Created: ${createdDate}</span>
                    </div>
                </div>
                <div class="text-teal-600">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                    </svg>
                </div>
            </div>
        `;
        return card;
    }

    /**
     * Update the visual selection of datasets
     */
    updateDatasetSelection(selectedDataset) {
        document.querySelectorAll('#datasetList .dataset-card').forEach(card => {
            card.classList.remove('selected');
        });
        const cards = document.querySelectorAll('#datasetList .dataset-card');
        cards.forEach(card => {
            if (card.querySelector('.dataset-name').textContent === (selectedDataset.name || selectedDataset.dataset_name)) {
                card.classList.add('selected');
            }
        });
    }

    /**
     * Load attributes and unique values for selected dataset
     */
    async loadAttributesAndValues(datasetId) {
        try {
            const response = await fetch(`${this.API_BASE_URL}/marketplace/datasets/${datasetId}/info`);
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to load attributes');
            }
            this.attributeTypes = data.queryable_attributes || {};
            this.sensitiveAttribute = data.sensitive_attribute;

            const valResp = await fetch(`${this.API_BASE_URL}/marketplace/datasets/${datasetId}/attribute-values`);
            const valData = await valResp.json();
            if (!valResp.ok) {
                throw new Error(valData.error || 'Failed to load attribute values');
            }
            this.attributeUniqueValues = valData.unique_values || {};
            this.attributeMinMax = valData.min_max || {};
            this.displayAttributes(this.attributeTypes);
            this.filters = [];
            this.renderFilters();
        } catch (error) {
            this.showError(`Failed to load attributes/values: ${error.message}`);
        }
    }

    /**
     * Display attributes and automatically select them
     */
    displayAttributes(attributes) {
        const container = document.getElementById('attributeGrid');
        container.innerHTML = '';
        this.selectedAttributes = [];
        Object.entries(attributes).forEach(([attrName, attrType]) => {
            const div = this.createAttributeItem(attrName, attrType);
            container.appendChild(div);
            this.selectedAttributes.push(attrName);
        });
        this.updateSelectedAttributes();
    }

    /**
     * Create an attribute item with only filters (no checkboxes)
     */
    createAttributeItem(attrName, attrType) {
        const div = document.createElement('div');
        div.className = 'attribute-item';
        const labelDiv = document.createElement('div');
        labelDiv.className = 'attribute-label';
        labelDiv.textContent = `${attrName} (${attrType})`;
        div.appendChild(labelDiv);
        const filterDiv = document.createElement('div');
        filterDiv.className = 'attribute-filter';
        if (attrType === 'categorical' && this.attributeUniqueValues[attrName]) {
            const select = this.createCategoricalFilter(attrName);
            filterDiv.appendChild(select);
        } else if (attrType === 'numeric' && this.attributeMinMax[attrName]) {
            const numericFilters = this.createNumericFilters(attrName);
            filterDiv.appendChild(numericFilters);
        }
        div.appendChild(filterDiv);
        return div;
    }

    /**
     * Create categorical filter dropdown
     */
    createCategoricalFilter(attrName) {
        const select = document.createElement('select');
        select.innerHTML = '<option value="">Filter...</option>';
        this.attributeUniqueValues[attrName].forEach(val => {
            const opt = document.createElement('option');
            opt.value = val;
            opt.textContent = val;
            select.appendChild(opt);
        });
        select.onchange = function () {
            this.setOrRemoveFilter(attrName, '=', select.value);
        }.bind(this);
        return select;
    }

    /**
     * Create numeric filter inputs
     */
    createNumericFilters(attrName) {
        const container = document.createElement('div');
        const minInput = document.createElement('input');
        minInput.type = 'number';
        minInput.placeholder = `Min (${this.attributeMinMax[attrName].min})`;
        minInput.onchange = function () {
            this.setOrRemoveFilter(attrName, '>=', minInput.value, true);
        }.bind(this);
        const maxInput = document.createElement('input');
        maxInput.type = 'number';
        maxInput.placeholder = `Max (${this.attributeMinMax[attrName].max})`;
        maxInput.onchange = function () {
            this.setOrRemoveFilter(attrName, '<=', maxInput.value, true);
        }.bind(this);
        container.appendChild(minInput);
        container.appendChild(maxInput);
        return container;
    }

    /**
     * Update selected attributes display
     */
    updateSelectedAttributes() {
        const display = document.getElementById('selectedAttributesDisplay');
        if (display) {
            if (this.selectedAttributes.length > 0) {
                display.textContent = 'All queryable attributes selected';
            } else {
                display.textContent = 'None';
            }
        }
        this.debouncedQueryPreview();
    }

    /**
     * Add a new filter row
     */
    addFilter() {
        this.filters.push({ attr: '', op: '=', value: '' });
        this.renderFilters();
        this.debouncedQueryPreview();
    }

    /**
     * Render all filter rows
     */
    renderFilters() {
        const container = document.getElementById('filtersContainer');
        if (!container) return;
        container.innerHTML = '';
        if (Object.keys(this.attributeTypes).length === 0) return;
        this.filters.forEach((filter, idx) => {
            const row = this.createFilterRow(filter, idx);
            container.appendChild(row);
        });
        this.debouncedAvailableRecordsUpdate();
        this.debouncedQueryPreview();
    }

    /**
     * Create a filter row element
     */
    createFilterRow(filter, idx) {
        const row = document.createElement('div');
        row.className = 'filter-row';
        const attrSelect = this.createAttributeSelect(filter.attr);
        attrSelect.onchange = e => {
            filter.attr = e.target.value;
            this.renderFilters();
            this.debouncedAvailableRecordsUpdate();
            this.debouncedQueryPreview();
        };
        const opSelect = this.createOperatorSelect(filter.attr, filter.op);
        opSelect.onchange = e => {
            filter.op = e.target.value;
            this.debouncedAvailableRecordsUpdate();
            this.debouncedQueryPreview();
        };
        const valInput = document.createElement('input');
        valInput.type = this.attributeTypes[filter.attr] === 'numeric' ? 'number' : 'text';
        valInput.value = filter.value;
        valInput.placeholder = 'Value';
        valInput.oninput = e => {
            filter.value = e.target.value;
            this.debouncedAvailableRecordsUpdate();
            this.debouncedQueryPreview();
        };
        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.textContent = 'Remove';
        removeBtn.onclick = () => {
            this.filters.splice(idx, 1);
            this.renderFilters();
            this.debouncedAvailableRecordsUpdate();
            this.debouncedQueryPreview();
        };
        row.appendChild(attrSelect);
        row.appendChild(opSelect);
        row.appendChild(valInput);
        row.appendChild(removeBtn);
        return row;
    }

    /**
     * Create attribute select dropdown
     */
    createAttributeSelect(selectedAttr) {
        const select = document.createElement('select');
        Object.keys(this.attributeTypes).forEach(attr => {
            const opt = document.createElement('option');
            opt.value = attr;
            opt.textContent = attr;
            if (selectedAttr === attr) opt.selected = true;
            select.appendChild(opt);
        });
        return select;
    }

    /**
     * Create operator select dropdown
     */
    createOperatorSelect(attr, selectedOp) {
        const select = document.createElement('select');
        const ops = this.attributeTypes[attr] === 'numeric' ?
            ['=', '!=', '>', '>=', '<', '<='] :
            ['=', '!=', 'contains'];
        ops.forEach(op => {
            const opt = document.createElement('option');
            opt.value = op;
            opt.textContent = op;
            if (selectedOp === op) opt.selected = true;
            select.appendChild(opt);
        });
        return select;
    }

    /**
     * Set or remove a filter
     */
    setOrRemoveFilter(attr, op, value, isNumericRange) {
        this.filters = this.filters.filter(f => !(f.attr === attr && (isNumericRange ? f.op === op : true)));
        if (value !== '') {
            this.filters.push({ attr, op, value });
        }
        this.renderFilters();
        this.debouncedAvailableRecordsUpdate();
        this.debouncedQueryPreview();
    }

    /**
     * Get price quote for selected configuration
     */
    async getPriceQuote() {
        if (this.selectedAttributes.length === 0) {
            this.showError('Please select at least one attribute');
            return;
        }
        const numValues = parseInt(document.getElementById('numValues').value);
        const epsilon = parseFloat(document.getElementById('epsilon').value);
        if (!numValues || numValues < 1) {
            this.showError('Please enter a valid number of records');
            return;
        }
        if (!epsilon || epsilon < 0.1 || epsilon > 1.0) {
            this.showError('Please enter a valid privacy level (0.1 to 1.0)');
            return;
        }
        try {
            const response = await fetch(`${this.API_BASE_URL}/marketplace/datasets/${this.selectedDatasetId}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    selected_attributes: this.selectedAttributes,
                    num_values_per_attribute: numValues,
                    epsilon: epsilon,
                    filters: this.filters.filter(f => f.attr && f.value !== '')
                })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get price quote');
            }
            this.displayPriceQuote(data);
            this.currentPriceQuote = data;
        } catch (error) {
            this.showError(`Failed to get price quote: ${error.message}`);
        }
    }

    /**
     * Display price quote in the UI
     */
    displayPriceQuote(quote) {
        const container = document.getElementById('priceDetails');

        let scarcityInfo = '';
        if (quote.scarcity_factor && quote.scarcity_factor > 1.0) {
            scarcityInfo = `
                <div class="scarcity-info">
                    <p><strong>Scarcity Factor:</strong> ${quote.scarcity_factor.toFixed(2)}x</p>
                    <p><strong>Filtered Records:</strong> ${quote.filtered_count} of ${quote.total_count} total</p>
                    <p class="text-sm text-orange-600">⚠️ Higher price due to limited filtered data</p>
                </div>
            `;
        }

        let priceBreakdown = '';
        if (quote.price_breakdown) {
            const baseMin = quote.price_breakdown.base_price_range?.[0] || 0;
            const baseMax = quote.price_breakdown.base_price_range?.[1] || 0;
            const adjustedMin = quote.price_breakdown.adjusted_price_range?.[0] || 0;
            const adjustedMax = quote.price_breakdown.adjusted_price_range?.[1] || 0;

            priceBreakdown = `
                <div class="price-breakdown">
                    <p><strong>Base Price Range:</strong> $${baseMin.toFixed(2)} - $${baseMax.toFixed(2)}</p>
                    <p><strong>Adjusted Price Range:</strong> $${adjustedMin.toFixed(2)} - $${adjustedMax.toFixed(2)}</p>
                </div>
            `;
        }

        container.innerHTML = `
            <div class="price-header">
                <h3>Price Quote</h3>
                <span class="price-amount">$${quote.final_price.toFixed(2)}</span>
            </div>
            <div class="price-info">
                <p><strong>Selected Attributes:</strong> ${this.selectedAttributes.join(', ')}</p>
                <p><strong>Records:</strong> ${document.getElementById('numValues').value}</p>
                <p><strong>Privacy Level:</strong> ${document.getElementById('epsilon').value}</p>
                <p><strong>Final Price Range:</strong> $${quote.p_min_k_for_query.toFixed(2)} - $${quote.p_max_k_for_query.toFixed(2)}</p>
                ${priceBreakdown}
                ${scarcityInfo}
            </div>
            

        `;
        this.showElement('priceQuote');
    }

    /**
     * Purchase the data
     */
    async purchaseData() {
        if (!this.currentPriceQuote) {
            this.showError('Please get a price quote first');
            return;
        }
        try {
            const response = await fetch(`${this.API_BASE_URL}/marketplace/purchase`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query_id: this.currentPriceQuote.query_id })
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || 'Purchase failed');
            }
            this.displayResults(data);
        } catch (error) {
            this.showError(`Purchase failed: ${error.message}`);
        }
    }

    /**
     * Display results in the UI
     */
    displayResults(data) {
        this.displayPrivacyNote(data);
        this.displayDataTable(data);
        // this.displayQueryIdInfo(data);
        this.displayQueryComparisonReport(data);
        this.showElement('results');
    }


    /**
     * Display privacy note
     */
    displayPrivacyNote(data) {
        const privacyContainer = document.getElementById('privacyNote');
        if (data.data_response && data.data_response.privacy_note) {
            privacyContainer.innerHTML = `
                <div class="privacy-note-content">
                    <span>🔒</span>
                    <p>${data.data_response.privacy_note}</p>
                </div>
            `;
            this.showElement('privacyNote');
        }
    }

    /**
     * Display data table with enhanced features
     */
    displayDataTable(data) {
        const tableContainer = document.getElementById('dataTable');
        let rows = [];
        if (data.data_response && data.data_response.privatized_data && data.data_response.privatized_data.length > 0) {
            // Unwrap if data is [{data: {...}}, ...]
            if (data.data_response.privatized_data[0].data) {
                rows = data.data_response.privatized_data.map(row => row.data);
            } else {
                rows = data.data_response.privatized_data;
            }

            // Store data for table management
            this.dataTableRows = rows;
            this.dataTablePage = 1;
            this.dataTablePageSize = 20;
            this.dataTableSortColumn = null;
            this.dataTableSortDirection = 'asc';
            this.dataTableFilterText = '';

            this.renderEnhancedTable();

            // Show raw and privatized counts if available
            const countsDiv = document.getElementById('dataCounts');
            if (countsDiv) {
                const rawCount = data.data_response.raw_count;
                const privatizedCount = data.data_response.privatized_count;
                if (typeof rawCount !== 'undefined' && typeof privatizedCount !== 'undefined') {
                    countsDiv.innerHTML = `
                        <div><strong>Raw Data Count:</strong> ${rawCount}</div>
                        <div><strong>Privatized Data Count:</strong> ${privatizedCount}</div>
                    `;
                } else {
                    countsDiv.innerHTML = '';
                }
            }
        } else {
            tableContainer.innerHTML = `
                <div class="no-data-message">
                    <div class="no-data-icon">📊</div>
                    <h3>No Data Available</h3>
                    <p>The query returned no results. Try adjusting your filters or privacy settings.</p>
                </div>
            `;
        }
    }

    /**
     * Render enhanced table with sorting, filtering, and pagination
     */
    renderEnhancedTable() {
        const tableContainer = document.getElementById('dataTable');
        const rows = this.dataTableRows || [];

        if (rows.length === 0) {
            tableContainer.innerHTML = '<div class="no-data-message">No data to display</div>';
            return;
        }

        // Apply filtering
        let filteredRows = this.applyTableFilter(rows);

        // Apply sorting
        filteredRows = this.applyTableSort(filteredRows);

        // Apply pagination
        const page = this.dataTablePage || 1;
        const pageSize = this.dataTablePageSize || 20;
        const totalPages = Math.ceil(filteredRows.length / pageSize);
        const startIdx = (page - 1) * pageSize;
        const endIdx = Math.min(startIdx + pageSize, filteredRows.length);
        const pageRows = filteredRows.slice(startIdx, endIdx);

        // Create table container
        const tableWrapper = document.createElement('div');
        tableWrapper.className = 'enhanced-table-wrapper';

        // Create table header with controls
        const tableHeader = this.createTableHeader(rows[0], filteredRows.length, totalPages);
        tableWrapper.appendChild(tableHeader);

        // Create scrollable table
        const scrollDiv = document.createElement('div');
        scrollDiv.className = 'enhanced-table-scroll';
        scrollDiv.appendChild(this.createEnhancedTable(pageRows));
        tableWrapper.appendChild(scrollDiv);

        // Create pagination
        const pagination = this.createEnhancedPagination(page, totalPages, filteredRows.length);
        tableWrapper.appendChild(pagination);

        // Clear and append
        tableContainer.innerHTML = '';
        tableContainer.appendChild(tableWrapper);
    }

    /**
     * Create table header with controls
     */
    createTableHeader(firstRow, totalRows, totalPages) {
        const header = document.createElement('div');
        header.className = 'table-header-controls';

        const columns = Object.keys(firstRow);

        // Table info
        const infoDiv = document.createElement('div');
        infoDiv.className = 'table-info';
        infoDiv.innerHTML = `
            <div class="table-stats">
                <span class="stat-item">
                    <span class="stat-label">📊 Records:</span>
                    <span class="stat-value">${totalRows}</span>
                </span>
                <span class="stat-item">
                    <span class="stat-label">📋 Columns:</span>
                    <span class="stat-value">${columns.length}</span>
                </span>
                <span class="stat-item">
                    <span class="stat-label">📄 Page:</span>
                    <span class="stat-value">${this.dataTablePage} of ${totalPages}</span>
                </span>
            </div>
        `;

        // Search and controls
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'table-controls';

        // Search input
        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = '🔍 Search in table...';
        searchInput.className = 'table-search';
        searchInput.value = this.dataTableFilterText || '';
        searchInput.oninput = (e) => {
            this.dataTableFilterText = e.target.value;
            this.dataTablePage = 1; // Reset to first page
            this.renderEnhancedTable();
        };

        // Export button
        const exportBtn = document.createElement('button');
        exportBtn.className = 'btn btn-secondary btn-sm';
        exportBtn.innerHTML = '📥 Export CSV';
        exportBtn.onclick = () => this.exportTableToCSV();

        // Page size selector
        const pageSizeSelect = document.createElement('select');
        pageSizeSelect.className = 'page-size-select';
        [10, 20, 50, 100].forEach(size => {
            const option = document.createElement('option');
            option.value = size;
            option.textContent = `${size} per page`;
            if (size === this.dataTablePageSize) option.selected = true;
            pageSizeSelect.appendChild(option);
        });
        pageSizeSelect.onchange = (e) => {
            this.dataTablePageSize = parseInt(e.target.value);
            this.dataTablePage = 1; // Reset to first page
            this.renderEnhancedTable();
        };

        controlsDiv.appendChild(searchInput);
        controlsDiv.appendChild(pageSizeSelect);
        controlsDiv.appendChild(exportBtn);

        header.appendChild(infoDiv);
        header.appendChild(controlsDiv);
        return header;
    }

    /**
     * Create enhanced table with sorting
     */
    createEnhancedTable(data) {
        if (!data || data.length === 0) return document.createElement('div');

        const table = document.createElement('table');
        table.className = 'enhanced-data-table';

        const headers = Object.keys(data[0]);

        // Create header row with sorting
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');

        headers.forEach(header => {
            const th = document.createElement('th');
            th.className = 'sortable-header';

            const headerContent = document.createElement('div');
            headerContent.className = 'header-content';

            const headerText = document.createElement('span');
            headerText.textContent = header;
            headerText.className = 'header-text';

            const sortIcon = document.createElement('span');
            sortIcon.className = 'sort-icon';
            sortIcon.innerHTML = this.getSortIcon(header);

            headerContent.appendChild(headerText);
            headerContent.appendChild(sortIcon);
            th.appendChild(headerContent);

            // Add click handler for sorting
            th.onclick = () => {
                this.toggleTableSort(header);
                this.renderEnhancedTable();
            };

            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create body
        const tbody = document.createElement('tbody');
        data.forEach((row, rowIndex) => {
            const tr = document.createElement('tr');
            tr.className = rowIndex % 2 === 0 ? 'even-row' : 'odd-row';

            headers.forEach(header => {
                const td = document.createElement('td');
                let value = row[header] || '';

                // Format value based on type
                value = this.formatTableCellValue(value);
                td.innerHTML = value;

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        return table;
    }

    /**
     * Format table cell value
     */
    formatTableCellValue(value) {
        if (value === null || value === undefined || value === '') {
            return '<span class="empty-value">—</span>';
        }

        // If value is a number, format it
        if (!isNaN(value) && value !== '' && value !== null) {
            const num = parseFloat(value);
            if (!isNaN(num)) {
                // Format based on magnitude
                if (Math.abs(num) >= 1000) {
                    return `<span class="numeric-value">${num.toLocaleString()}</span>`;
                } else if (Math.abs(num) < 1 && num !== 0) {
                    return `<span class="numeric-value">${num.toFixed(4)}</span>`;
                } else {
                    return `<span class="numeric-value">${num.toFixed(2)}</span>`;
                }
            }
        }

        // For text values, truncate if too long
        const strValue = String(value);
        if (strValue.length > 50) {
            return `<span class="text-value" title="${strValue}">${strValue.substring(0, 47)}...</span>`;
        }

        return `<span class="text-value">${strValue}</span>`;
    }

    /**
     * Get sort icon for header
     */
    getSortIcon(column) {
        if (this.dataTableSortColumn !== column) {
            return '↕️';
        }
        return this.dataTableSortDirection === 'asc' ? '↑' : '↓';
    }

    /**
     * Toggle table sorting
     */
    toggleTableSort(column) {
        if (this.dataTableSortColumn === column) {
            this.dataTableSortDirection = this.dataTableSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.dataTableSortColumn = column;
            this.dataTableSortDirection = 'asc';
        }
    }

    /**
     * Apply table filtering
     */
    applyTableFilter(rows) {
        if (!this.dataTableFilterText) return rows;

        const filterText = this.dataTableFilterText.toLowerCase();
        return rows.filter(row => {
            return Object.values(row).some(value =>
                String(value).toLowerCase().includes(filterText)
            );
        });
    }

    /**
     * Apply table sorting
     */
    applyTableSort(rows) {
        if (!this.dataTableSortColumn) return rows;

        return rows.sort((a, b) => {
            const aVal = a[this.dataTableSortColumn];
            const bVal = b[this.dataTableSortColumn];

            // Handle null/undefined values
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;

            // Try numeric comparison first
            const aNum = parseFloat(aVal);
            const bNum = parseFloat(bVal);

            if (!isNaN(aNum) && !isNaN(bNum)) {
                return this.dataTableSortDirection === 'asc' ? aNum - bNum : bNum - aNum;
            }

            // String comparison
            const aStr = String(aVal).toLowerCase();
            const bStr = String(bVal).toLowerCase();

            if (this.dataTableSortDirection === 'asc') {
                return aStr.localeCompare(bStr);
            } else {
                return bStr.localeCompare(aStr);
            }
        });
    }

    /**
     * Create enhanced pagination
     */
    createEnhancedPagination(currentPage, totalPages, totalRows) {
        const pagination = document.createElement('div');
        pagination.className = 'enhanced-pagination';

        if (totalPages <= 1) {
            pagination.innerHTML = `<div class="pagination-info">Showing all ${totalRows} records</div>`;
            return pagination;
        }

        // Previous button
        const prevBtn = document.createElement('button');
        prevBtn.className = 'pagination-btn';
        prevBtn.innerHTML = '← Previous';
        prevBtn.disabled = currentPage === 1;
        prevBtn.onclick = () => {
            this.dataTablePage--;
            this.renderEnhancedTable();
        };

        // Next button
        const nextBtn = document.createElement('button');
        nextBtn.className = 'pagination-btn';
        nextBtn.innerHTML = 'Next →';
        nextBtn.disabled = currentPage === totalPages;
        nextBtn.onclick = () => {
            this.dataTablePage++;
            this.renderEnhancedTable();
        };

        // Page numbers
        const pageNumbers = document.createElement('div');
        pageNumbers.className = 'page-numbers';

        const startPage = Math.max(1, currentPage - 2);
        const endPage = Math.min(totalPages, currentPage + 2);

        for (let i = startPage; i <= endPage; i++) {
            const pageBtn = document.createElement('button');
            pageBtn.className = `page-number ${i === currentPage ? 'active' : ''}`;
            pageBtn.textContent = i;
            pageBtn.onclick = () => {
                this.dataTablePage = i;
                this.renderEnhancedTable();
            };
            pageNumbers.appendChild(pageBtn);
        }

        // Info
        const info = document.createElement('div');
        info.className = 'pagination-info';
        const startRow = (currentPage - 1) * this.dataTablePageSize + 1;
        const endRow = Math.min(currentPage * this.dataTablePageSize, totalRows);
        info.textContent = `Showing ${startRow}-${endRow} of ${totalRows} records`;

        pagination.appendChild(prevBtn);
        pagination.appendChild(pageNumbers);
        pagination.appendChild(nextBtn);
        pagination.appendChild(info);

        return pagination;
    }

    /**
     * Export table to CSV
     */
    exportTableToCSV() {
        const rows = this.dataTableRows || [];
        if (rows.length === 0) {
            this.showError('No data to export');
            return;
        }

        try {
            const headers = Object.keys(rows[0]);
            const csvContent = [
                headers.join(','),
                ...rows.map(row =>
                    headers.map(header => {
                        const value = row[header] || '';
                        // Escape commas and quotes
                        const escaped = String(value).replace(/"/g, '""');
                        return `"${escaped}"`;
                    }).join(',')
                )
            ].join('\n');

            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', `data_export_${new Date().toISOString().split('T')[0]}.csv`);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Show success message
            this.showSuccessMessage('Data exported successfully!');
        } catch (error) {
            this.showError(`Export failed: ${error.message}`);
        }
    }

    /**
     * Show success message
     */
    showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = `
            <div class="success-content">
                <span class="success-icon">✅</span>
                <span class="success-text">${message}</span>
            </div>
        `;

        document.body.appendChild(successDiv);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (successDiv.parentNode) {
                successDiv.parentNode.removeChild(successDiv);
            }
        }, 3000);
    }

    /**
     * Populate the target variable dropdown
     */
    populateTargetVariableDropdown() {
        const select = document.getElementById('targetVariable');
        if (!select) return;
        select.innerHTML = '';

        // --- Start of Modified Section ---

        // Create a Set to hold all unique attribute names
        const allAttributes = new Set(Object.keys(this.attributeTypes));

        // Add the sensitive attribute if it exists
        if (this.sensitiveAttribute) {
            allAttributes.add(this.sensitiveAttribute);
        }

        // Populate the dropdown from the Set
        allAttributes.forEach(attr => {
            const option = document.createElement('option');
            option.value = attr;
            option.textContent = attr;
            select.appendChild(option);
        });

        // --- End of Modified Section ---
    }

    /**
     * Display the query comparison report under the data table
     */
    async displayQueryComparisonReport(data) {
        console.log("==================?>>>>>>>data",data);
        const reportContainer = document.getElementById('queryComparisonReport');
        reportContainer.innerHTML = '';
        if (!data || !data.query_id || !this.selectedDatasetId) return;

        // Use the data already available in the purchase response
        const dataResponse = data.data_response;
        if (!dataResponse) {
            reportContainer.innerHTML = '<div class="comparison-report-card">No data response available for comparison.</div>';
            return;
        }

        const raw = dataResponse.raw_count || 0;
        const noisy = dataResponse.privatized_count || 0;
        const epsilon = dataResponse.epsilon || parseFloat(document.getElementById('epsilon')?.value) || 0.5;
        
        // Build query string from filters
        let queryStr = "All records";
        if (dataResponse.filters && dataResponse.filters.length > 0) {
            const filter = dataResponse.filters[0];
            queryStr = `${filter.attr} ${filter.op} ${filter.value}`;
        } else if (this.filters && this.filters.length > 0) {
            const filter = this.filters[0];
            queryStr = `${filter.attr} ${filter.op} ${filter.value}`;
        }

        const diff = Math.abs(raw - noisy);
        const errorPct = raw > 0 ? ((diff / raw) * 100).toFixed(2) : '0.00';
        
        reportContainer.innerHTML = `
            <section class="card" style="margin-top:1.5rem;">
                <h2>📊 Query Comparison Report</h2>
                <div class="comparison-metrics" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1rem;margin:1rem 0;">
                    <div class="metric-card"><div class="metric-value">${raw}</div><div class="metric-label">Raw Count</div></div>
                    <div class="metric-card"><div class="metric-value">${noisy}</div><div class="metric-label">Noisy Count</div></div>
                    <div class="metric-card"><div class="metric-value">${diff}</div><div class="metric-label">Difference</div></div>
                    <div class="metric-card"><div class="metric-value">${errorPct}%</div><div class="metric-label">Error %</div></div>
                </div>
                <div class="chart-container" style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:1rem;height:300px;">
                    <canvas id="comparisonChartConsumer"></canvas>
                </div>
                <div style="margin-top:1rem;font-size:0.95rem;color:#555;">Query: <code>${queryStr}</code> | Epsilon: <b>${epsilon}</b></div>
            </section>
        `;
        
        // Render chart
        setTimeout(() => {
            const ctx = document.getElementById('comparisonChartConsumer');
            if (ctx) {
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: ['Raw', `Noisy (ε=${epsilon})`],
                        datasets: [{
                            label: 'Count',
                            data: [raw, noisy],
                            backgroundColor: ['#10b981', '#3b82f6'],
                            borderColor: ['#059669', '#2563eb'],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { y: { beginAtZero: true, title: { display: true, text: 'Count' } } },
                        plugins: { title: { display: true, text: 'Query Results: Raw vs Noisy' }, legend: { display: false } }
                    }
                });
            }
        }, 100);
    }

    // Utility methods
    showElement(id) {
        const element = document.getElementById(id);
        if (element) {
            element.classList.remove('hidden');
        } else {
            console.error(`Element with id '${id}' not found`);
        }
    }
    hideElement(id) {
        const element = document.getElementById(id);
        if (element) element.classList.add('hidden');
    }
    showLoading(id) {
        const element = document.getElementById(id);
        if (element) element.classList.remove('hidden');
    }
    hideLoading(id) {
        const element = document.getElementById(id);
        if (element) element.classList.add('hidden');
    }
    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.textContent = message;
            this.showElement('errorDisplay');
            setTimeout(() => this.hideElement('errorDisplay'), 5000);
        }
    }
    formatDate(timestamp) {
        if (!timestamp) return 'Unknown';
        try {
            return new Date(timestamp).toLocaleDateString();
        } catch {
            return 'Unknown';
        }
    }
}

// Initialize the interface when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    new ConsumerInterface();
});

window.ConsumerInterface = ConsumerInterface;