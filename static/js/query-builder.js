// Smart Query Builder JavaScript

class SmartQueryBuilder {
    constructor(containerId) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        this.dataset = null;
        this.attributes = {};
        this.querySteps = [];
        this.templates = this.getQueryTemplates();
        this.simpleQuery = '';
        this.init();
    }

    init() {
        this.render();
        this.bindEvents();
    }

    render() {
        this.container.innerHTML = `
            <div class="query-builder-container">
                <!-- Simple Query Input -->
                <div class="simple-query-box" style="position:relative;">
                    <label for="simpleQueryInput"><strong>Simple Query:</strong> <span style="font-weight:normal">(e.g. age = 25, name = John)</span></label>
                    <textarea id="simpleQueryInput" rows="2" placeholder="age = 25, name = John" style="width:100%;margin-bottom:10px;"></textarea>
                    <div id="autocompleteDropdown" class="autocomplete-dropdown" style="display:none;position:absolute;z-index:10;left:0;right:0;"></div>
                </div>
                <div class="builder-toolbar">
                    <button class="btn btn-secondary" id="addStepBtn">+ Add Step</button>
                    <button class="btn btn-primary" id="executeQueryBtn">▶️ Execute Query</button>
                    <button class="btn btn-primary" id="executeSimpleQueryBtn">▶️ Execute Simple Query</button>
                </div>
                <div class="query-feedback" id="queryFeedback" style="display: none;"></div>
            </div>
        `;
    }

    renderTemplates() {
        return this.templates.map(template => `
            <div class="template-card" data-template="${template.id}">
                <h5>${template.name}</h5>
                <p>${template.description}</p>
                <div class="template-tags">
                    ${template.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
        `).join('');
    }

    getQueryTemplates() {
        return [
            {
                id: 'basic_select',
                name: 'Basic Selection',
                description: 'Select specific attributes from the dataset',
                tags: ['Basic', 'Selection'],
                steps: [
                    { type: 'select', attributes: [] }
                ]
            },
            {
                id: 'filtered_query',
                name: 'Filtered Query',
                description: 'Select attributes with filtering conditions',
                tags: ['Filter', 'Conditional'],
                steps: [
                    { type: 'select', attributes: [] },
                    { type: 'filter', conditions: [] }
                ]
            },
            {
                id: 'aggregation',
                name: 'Aggregation',
                description: 'Group and aggregate data with privacy protection',
                tags: ['Aggregate', 'Group'],
                steps: [
                    { type: 'select', attributes: [] },
                    { type: 'group', by: [] },
                    { type: 'aggregate', functions: [] }
                ]
            },
            {
                id: 'privacy_focused',
                name: 'Privacy-Focused',
                description: 'High privacy settings with minimal data exposure',
                tags: ['Privacy', 'Secure'],
                steps: [
                    { type: 'select', attributes: [] },
                    { type: 'privacy', epsilon: 0.1 }
                ]
            }
        ];
    }

    bindEvents() {
        // Template selection
        this.container.addEventListener('click', (e) => {
            if (e.target.closest('.template-card')) {
                const templateId = e.target.closest('.template-card').dataset.template;
                this.loadTemplate(templateId);
            }
        });

        // Builder buttons
        this.container.addEventListener('click', (e) => {
            if (e.target.id === 'addStepBtn') {
                this.addQueryStep();
            } else if (e.target.id === 'executeQueryBtn') {
                this.executeQuery();
            } else if (e.target.id === 'clearQueryBtn') {
                this.clearQuery();
            }
        });

        // Step-specific events
        this.container.addEventListener('change', (e) => {
            if (e.target.closest('.query-step')) {
                this.updateQueryStep(e.target.closest('.query-step'));
            }
        });

        this.container.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-step')) {
                const stepElement = e.target.closest('.query-step');
                this.removeQueryStep(stepElement);
            }
        });

        // Simple query execution
        this.container.querySelector('#executeSimpleQueryBtn').addEventListener('click', () => {
            const input = this.container.querySelector('#simpleQueryInput').value;
            this.simpleQuery = input;
            const filters = this.parseSimpleQuery(input);
            if (filters === null) {
                this.showFeedback('❌ Invalid query format. Use: age = 25, name = John', true);
                this.showParsedFilters([]);
                return;
            }
            // Attribute validation
            const invalidAttrs = filters.filter(f => !(f.attribute in this.attributes)).map(f => f.attribute);
            if (invalidAttrs.length > 0) {
                this.showFeedback('❌ Invalid attribute(s): ' + invalidAttrs.join(', '), true);
                this.showParsedFilters([]);
                return;
            }
            this.showFeedback('');
            this.showParsedFilters(filters);
            this.executeSimpleQuery(filters);
        });

        // Auto-complete for attributes and operators
        const inputBox = this.container.querySelector('#simpleQueryInput');
        const dropdown = this.container.querySelector('#autocompleteDropdown');
        inputBox.addEventListener('input', (e) => {
            this.handleAutocomplete(e.target, dropdown);
        });
        inputBox.addEventListener('keydown', (e) => {
            if (dropdown.style.display === 'block') {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    this.moveAutocompleteSelection(1, dropdown);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    this.moveAutocompleteSelection(-1, dropdown);
                } else if (e.key === 'Enter' || e.key === 'Tab') {
                    if (this.autocompleteActiveIndex !== undefined) {
                        e.preventDefault();
                        this.selectAutocomplete(dropdown, inputBox);
                    }
                } else if (e.key === 'Escape') {
                    dropdown.style.display = 'none';
                }
            }
        });
        document.addEventListener('click', (e) => {
            if (!this.container.contains(e.target)) {
                dropdown.style.display = 'none';
            }
        });

        // Filter attribute change
        this.container.addEventListener('change', async (e) => {
            if (e.target.classList.contains('filter-attr')) {
                const stepElement = e.target.closest('.query-step');
                const stepId = parseInt(stepElement.dataset.stepId);
                const step = this.querySteps.find(s => s.id === stepId);
                if (!step) return;
                const index = parseInt(e.target.getAttribute('data-index'));
                const attr = e.target.value;
                // Fetch unique values for this attribute
                const datasetId = window.selectedDatasetId;
                const res = await fetch(`/marketplace/datasets/${datasetId}/attribute-values`);
                const data = await res.json();
                const valueOptions = (data.unique_values && data.unique_values[attr]) ? data.unique_values[attr] : [];
                // Update the condition's valueOptions and reset value
                step.config.conditions[index].valueOptions = valueOptions;
                step.config.conditions[index].value = valueOptions[0] || '';
                this.renderQuerySteps();
            }
        });
    }

    setDataset(dataset, attributes) {
        this.dataset = dataset;
        this.attributes = attributes;
        this.updateBuilderState();
    }

    updateBuilderState() {
        const addStepBtn = this.container.querySelector('#addStepBtn');
        const executeBtn = this.container.querySelector('#executeQueryBtn');
        const emptyState = this.container.querySelector('.empty-state');

        if (this.dataset) {
            addStepBtn.disabled = false;
            emptyState.innerHTML = `
                <p>🎯 Dataset: <strong>${this.dataset.name}</strong></p>
                <p class="text-sm text-gray-500">Start building your query using templates or add steps manually</p>
            `;
        } else {
            addStepBtn.disabled = true;
            executeBtn.disabled = true;
        }
    }

    loadTemplate(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;

        this.clearQuery();
        this.querySteps = [...template.steps];
        this.renderQuerySteps();
        this.updateQueryPreview();
    }

    addQueryStep() {
        if (!this.dataset) {
            alert('Please select a dataset first');
            return;
        }

        const stepTypes = [
            { value: 'select', label: 'Select Attributes' },
            { value: 'filter', label: 'Filter Data' },
            { value: 'group', label: 'Group By' },
            { value: 'aggregate', label: 'Aggregate' },
            { value: 'privacy', label: 'Privacy Settings' }
        ];

        const stepType = prompt('Select step type:\n' + 
            stepTypes.map((t, i) => `${i + 1}. ${t.label}`).join('\n'));

        if (!stepType) return;

        const stepIndex = parseInt(stepType) - 1;
        if (stepIndex >= 0 && stepIndex < stepTypes.length) {
            const newStep = {
                id: Date.now(),
                type: stepTypes[stepIndex].value,
                config: this.getDefaultConfig(stepTypes[stepIndex].value)
            };
            this.querySteps.push(newStep);
            this.renderQuerySteps();
            this.updateQueryPreview();
        }
    }

    getDefaultConfig(stepType) {
        switch (stepType) {
            case 'select':
                return { attributes: [] };
            case 'filter':
                return { conditions: [] };
            case 'group':
                return { by: [] };
            case 'aggregate':
                return { functions: [] };
            case 'privacy':
                return { epsilon: 0.5, delta: 0.0001 };
            default:
                return {};
        }
    }

    renderQuerySteps() {
        const container = this.container.querySelector('#queryStepsContainer');
        
        if (this.querySteps.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>🎯 Select a dataset first, then start building your query</p>
                    <p class="text-sm text-gray-500">Use templates or build from scratch</p>
                </div>
            `;
            return;
        }

        container.innerHTML = this.querySteps.map((step, index) => `
            <div class="query-step" data-step-id="${step.id}">
                <div class="step-header">
                    <span class="step-number">${index + 1}</span>
                    <span class="step-type">${this.getStepTypeLabel(step.type)}</span>
                    <button class="remove-step" title="Remove step">❌</button>
                </div>
                <div class="step-content">
                    ${this.renderStepContent(step)}
                </div>
            </div>
        `).join('');
    }

    getStepTypeLabel(type) {
        const labels = {
            'select': 'Select Attributes',
            'filter': 'Filter Data',
            'group': 'Group By',
            'aggregate': 'Aggregate',
            'privacy': 'Privacy Settings'
        };
        return labels[type] || type;
    }

    renderStepContent(step) {
        switch (step.type) {
            case 'select':
                return this.renderSelectStep(step);
            case 'filter':
                return this.renderFilterStep(step);
            case 'group':
                return this.renderGroupStep(step);
            case 'aggregate':
                return this.renderAggregateStep(step);
            case 'privacy':
                return this.renderPrivacyStep(step);
            default:
                return '<p>Unknown step type</p>';
        }
    }

    renderSelectStep(step) {
        // Remove attribute checkboxes, just show a message or nothing
        return `<div class="select-step-info">All columns (except sensitive) will be returned by default. You do not need to select attributes.</div>`;
    }

    renderFilterStep(step) {
        const attributes = Object.keys(this.attributes);
        return `
            <div class="filter-builder">
                <div class="filter-conditions">
                    ${step.config.conditions.map((condition, index) => `
                        <div class="filter-condition">
                            <select class="filter-attr" data-index="${index}">
                                ${attributes.map(attr => `
                                    <option value="${attr}" ${condition.attribute === attr ? 'selected' : ''}>
                                        ${attr}
                                    </option>
                                `).join('')}
                            </select>
                            <select class="filter-op">
                                <option value="=" ${condition.operator === '=' ? 'selected' : ''}>=</option>
                                <option value="!=" ${condition.operator === '!=' ? 'selected' : ''}>!=</option>
                                <option value=">" ${condition.operator === '>' ? 'selected' : ''}>></option>
                                <option value="<" ${condition.operator === '<' ? 'selected' : ''}><</option>
                                <option value=">=" ${condition.operator === '>=' ? 'selected' : ''}>>=</option>
                                <option value="<=" ${condition.operator === '<=' ? 'selected' : ''}><=</option>
                            </select>
                            <select class="filter-value" data-index="${index}">
                                ${Array.isArray(condition.valueOptions) ? condition.valueOptions.map(val => `
                                    <option value="${val}" ${condition.value == val ? 'selected' : ''}>${val}</option>
                                `).join('') : `<option value="">Select value</option>`}
                            </select>
                            <button class="remove-condition" data-index="${index}">❌</button>
                        </div>
                    `).join('')}
                </div>
                <button class="add-condition">➕ Add Condition</button>
            </div>
        `;
    }

    renderGroupStep(step) {
        const attributes = Object.keys(this.attributes);
        return `
            <div class="group-builder">
                <label>Group by attributes:</label>
                <div class="attribute-grid">
                    ${attributes.map(attr => `
                        <label class="attribute-checkbox">
                            <input type="checkbox" value="${attr}" 
                                   ${step.config.by.includes(attr) ? 'checked' : ''}>
                            ${attr}
                        </label>
                    `).join('')}
                </div>
            </div>
        `;
    }

    renderAggregateStep(step) {
        return `
            <div class="aggregate-builder">
                <label>Aggregation functions:</label>
                <div class="aggregate-functions">
                    <label><input type="checkbox" value="count"> Count</label>
                    <label><input type="checkbox" value="sum"> Sum</label>
                    <label><input type="checkbox" value="avg"> Average</label>
                    <label><input type="checkbox" value="min"> Min</label>
                    <label><input type="checkbox" value="max"> Max</label>
                </div>
            </div>
        `;
    }

    renderPrivacyStep(step) {
        return `
            <div class="privacy-settings">
                <div class="privacy-control">
                    <label>Privacy Level (ε):</label>
                    <input type="range" min="0.1" max="1.0" step="0.1" 
                           value="${step.config.epsilon}" class="epsilon-slider">
                    <span class="epsilon-value">${step.config.epsilon}</span>
                    <small>Lower = more private, Higher = more accurate</small>
                </div>
                <div class="privacy-control">
                    <label>Number of Records (K):</label>
                    <input type="number" min="1" max="1000" 
                           value="${step.config.num_values_k || 10}" class="num-records">
                </div>
            </div>
        `;
    }

    updateQueryStep(stepElement) {
        const stepId = parseInt(stepElement.dataset.stepId);
        const step = this.querySteps.find(s => s.id === stepId);
        if (!step) return;

        // Update step configuration based on user input
        this.updateStepConfig(step, stepElement);
        this.updateQueryPreview();
        this.validateQuery();
    }

    updateStepConfig(step, stepElement) {
        switch (step.type) {
            case 'select':
                const checkboxes = stepElement.querySelectorAll('input[type="checkbox"]:checked');
                step.config.attributes = Array.from(checkboxes).map(cb => cb.value);
                break;
            case 'filter':
                // Update filter conditions
                break;
            case 'group':
                const groupCheckboxes = stepElement.querySelectorAll('input[type="checkbox"]:checked');
                step.config.by = Array.from(groupCheckboxes).map(cb => cb.value);
                break;
            case 'aggregate':
                const aggCheckboxes = stepElement.querySelectorAll('input[type="checkbox"]:checked');
                step.config.functions = Array.from(aggCheckboxes).map(cb => cb.value);
                break;
            case 'privacy':
                const epsilonSlider = stepElement.querySelector('.epsilon-slider');
                const numRecords = stepElement.querySelector('.num-records');
                if (epsilonSlider) {
                    step.config.epsilon = parseFloat(epsilonSlider.value);
                    stepElement.querySelector('.epsilon-value').textContent = step.config.epsilon;
                }
                if (numRecords) {
                    step.config.num_values_k = parseInt(numRecords.value);
                }
                break;
        }
    }

    removeQueryStep(stepElement) {
        const stepId = parseInt(stepElement.dataset.stepId);
        this.querySteps = this.querySteps.filter(s => s.id !== stepId);
        this.renderQuerySteps();
        this.updateQueryPreview();
    }

    clearQuery() {
        this.querySteps = [];
        this.renderQuerySteps();
        this.updateQueryPreview();
        this.hideQueryPreview();
        this.hideQueryFeedback();
    }

    updateQueryPreview() {
        if (this.querySteps.length === 0) {
            this.hideQueryPreview();
            return;
        }

        const preview = this.container.querySelector('#queryPreview');
        const content = this.container.querySelector('#previewContent');
        
        const previewText = this.generateQueryPreview();
        content.innerHTML = `
            <div class="preview-text">
                <pre>${previewText}</pre>
            </div>
            <div class="preview-summary">
                <p><strong>Steps:</strong> ${this.querySteps.length}</p>
                <p><strong>Attributes:</strong> ${this.getSelectedAttributesCount()}</p>
                <p><strong>Privacy Level:</strong> ${this.getPrivacyLevel()}</p>
            </div>
        `;
        
        preview.style.display = 'block';
    }

    generateQueryPreview() {
        let preview = 'SELECT ';
        
        // Find select step
        const selectStep = this.querySteps.find(s => s.type === 'select');
        if (selectStep && selectStep.config.attributes.length > 0) {
            preview += selectStep.config.attributes.join(', ');
        } else {
            preview += '*';
        }
        
        preview += '\nFROM dataset';
        
        // Add filters
        const filterStep = this.querySteps.find(s => s.type === 'filter');
        if (filterStep && filterStep.config.conditions.length > 0) {
            preview += '\nWHERE ' + filterStep.config.conditions.map(c => 
                `${c.attribute} ${c.operator} ${c.value}`
            ).join(' AND ');
        }
        
        // Add grouping
        const groupStep = this.querySteps.find(s => s.type === 'group');
        if (groupStep && groupStep.config.by.length > 0) {
            preview += '\nGROUP BY ' + groupStep.config.by.join(', ');
        }
        
        // Add privacy settings
        const privacyStep = this.querySteps.find(s => s.type === 'privacy');
        if (privacyStep) {
            preview += `\nWITH PRIVACY ε=${privacyStep.config.epsilon}`;
            if (privacyStep.config.num_values_k) {
                preview += `, K=${privacyStep.config.num_values_k}`;
            }
        }
        
        return preview;
    }

    getSelectedAttributesCount() {
        const selectStep = this.querySteps.find(s => s.type === 'select');
        return selectStep ? selectStep.config.attributes.length : 0;
    }

    getPrivacyLevel() {
        const privacyStep = this.querySteps.find(s => s.type === 'privacy');
        return privacyStep ? privacyStep.config.epsilon : 'Not set';
    }

    hideQueryPreview() {
        const preview = this.container.querySelector('#queryPreview');
        if (preview) preview.style.display = 'none';
    }

    validateQuery() {
        const messages = [];
        const suggestions = [];

        // Check if query has required steps
        if (this.querySteps.length === 0) {
            messages.push('⚠️ No query steps defined');
        }

        // Check select step
        const selectStep = this.querySteps.find(s => s.type === 'select');
        if (!selectStep) {
            messages.push('⚠️ No SELECT step found');
            suggestions.push('Add a SELECT step to choose attributes');
        } else if (selectStep.config.attributes.length === 0) {
            messages.push('⚠️ No attributes selected');
            suggestions.push('Select at least one attribute in the SELECT step');
        }

        // Check privacy settings
        const privacyStep = this.querySteps.find(s => s.type === 'privacy');
        if (!privacyStep) {
            suggestions.push('Consider adding privacy settings for differential privacy');
        }

        this.showQueryFeedback(messages, suggestions);
        
        // Enable/disable execute button
        const executeBtn = this.container.querySelector('#executeQueryBtn');
        executeBtn.disabled = messages.length > 0 || this.querySteps.length === 0;
    }

    showQueryFeedback(messages, suggestions) {
        const feedback = this.container.querySelector('#queryFeedback');
        const messagesDiv = this.container.querySelector('#validationMessages');
        const suggestionsDiv = this.container.querySelector('#suggestions');

        if (messages.length === 0 && suggestions.length === 0) {
            feedback.style.display = 'none';
            return;
        }

        messagesDiv.innerHTML = messages.map(msg => `<div class="message">${msg}</div>`).join('');
        suggestionsDiv.innerHTML = suggestions.map(sugg => `<div class="suggestion">💡 ${sugg}</div>`).join('');
        
        feedback.style.display = 'block';
    }

    hideQueryFeedback() {
        const feedback = this.container.querySelector('#queryFeedback');
        if (feedback) feedback.style.display = 'none';
    }

    executeQuery() {
        if (this.querySteps.length === 0) {
            alert('No query to execute');
            return;
        }

        // Validate query
        this.validateQuery();
        const executeBtn = this.container.querySelector('#executeQueryBtn');
        if (executeBtn.disabled) {
            alert('Please fix validation errors before executing');
            return;
        }

        // Build query object
        const query = this.buildQueryObject();
        
        // Emit custom event for integration
        const event = new CustomEvent('queryExecute', {
            detail: query
        });
        this.container.dispatchEvent(event);
    }

    buildQueryObject() {
        const query = {
            selected_attributes: [],
            filters: [],
            num_values_k: 10,
            dp_epsilon: 0.5
        };

        // Extract data from steps
        this.querySteps.forEach(step => {
            switch (step.type) {
                case 'select':
                    query.selected_attributes = step.config.attributes;
                    break;
                case 'filter':
                    query.filters = step.config.conditions;
                    break;
                case 'privacy':
                    query.dp_epsilon = step.config.epsilon;
                    query.num_values_k = step.config.num_values_k;
                    break;
            }
        });

        return query;
    }

    parseSimpleQuery(input) {
        // Split by comma, then by =, support quoted values
        if (!input.trim()) return [];
        const parts = input.split(',');
        const filters = [];
        for (let part of parts) {
            part = part.trim();
            if (!part) continue;
            // Match: attr op value (op can be =, !=, >, <, >=, <=)
            const match = part.match(/^([a-zA-Z0-9_ ]+)\s*(=|!=|>=|<=|>|<)\s*(.+)$/);
            if (!match) return null;
            let [, attribute, operator, value] = match;
            attribute = attribute.trim();
            value = value.trim();
            // Remove quotes if present
            if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
                value = value.slice(1, -1);
            }
            filters.push({ attribute, operator, value });
        }
        return filters;
    }

    showFeedback(msg, isError) {
        const feedback = this.container.querySelector('#queryFeedback');
        if (!msg) {
            feedback.style.display = 'none';
            feedback.textContent = '';
            return;
        }
        feedback.style.display = 'block';
        feedback.textContent = msg;
        feedback.style.color = isError ? 'red' : 'green';
    }

    /**
     * Execute simple query with enhanced feedback
     */
    async executeSimpleQuery(filters) {
        if (!window.selectedDatasetId) {
            this.showFeedback('❌ Please select a dataset first', true);
            return;
        }

        try {
            // Show execution feedback
            this.showFeedback('🚀 Executing query with record-based architecture...', false);
            
            // Build query data
            const queryData = {
                selected_attributes: this.getSelectedAttributes(),
                num_values_per_attribute: this.getNumValues(),
                epsilon: this.getPrivacyLevel(),
                filters: filters.map(f => ({
                    attr: f.attribute,
                    op: f.operator,
                    value: f.value
                }))
            };

            // Execute query
            const response = await fetch(`${CONFIG.API.BASE_URL}/marketplace/datasets/${window.selectedDatasetId}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(queryData)
            });

            if (response.ok) {
                const result = await response.json();
                this.showFeedback(`✅ Query executed successfully! Price: $${result.final_price?.toFixed(2) || 'N/A'}`, false);
                
                // Enhanced: Show performance metrics
                this.showPerformanceMetrics(result);
            } else {
                const error = await response.json();
                this.showFeedback(`❌ Query failed: ${error.error || 'Unknown error'}`, true);
            }
        } catch (error) {
            this.showFeedback(`❌ Network error: ${error.message}`, true);
        }
    }

    showPerformanceMetrics(result) {
        const metricsContainer = document.getElementById('performanceMetrics');
        if (!metricsContainer) {
            const feedbackDiv = document.getElementById('queryFeedback');
            const metricsDiv = document.createElement('div');
            metricsDiv.id = 'performanceMetrics';
            metricsDiv.className = 'performance-metrics bg-green-50 border border-green-200 rounded-md p-3 mt-3';
            feedbackDiv.appendChild(metricsDiv);
        }

        const container = document.getElementById('performanceMetrics');
        container.innerHTML = `
            <div class="text-sm text-green-800">
                <h5 class="font-medium mb-2">📊 Query Performance</h5>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div>💰 Price Range: $${result.p_min_k_for_query?.toFixed(2) || 'N/A'} - $${result.p_max_k_for_query?.toFixed(2) || 'N/A'}</div>
                    <div>🛡️ Privacy Level: ε = ${result.dp_epsilon_applied || 'N/A'}</div>
                    <div>⚡ Architecture: Record-based</div>
                    <div>�� Query ID: ${result.query_id?.substring(0, 8) || 'N/A'}...</div>
                </div>
            </div>
        `;
        container.style.display = 'block';
    }

    showParsedFilters(filters) {
        let summaryBox = this.container.querySelector('#parsedFiltersBox');
        if (!summaryBox) {
            summaryBox = document.createElement('div');
            summaryBox.id = 'parsedFiltersBox';
            summaryBox.style.margin = '10px 0';
            this.container.querySelector('.simple-query-box').appendChild(summaryBox);
        }
        if (!filters || filters.length === 0) {
            summaryBox.innerHTML = '';
            return;
        }
        summaryBox.innerHTML = '<strong>Parsed Filters:</strong> ' +
            filters.map(f => `<span class="parsed-filter">${f.attribute} ${f.operator} ${f.value}</span>`).join(' <b>&amp;</b> ');
    }

    handleAutocomplete(inputBox, dropdown) {
        const value = inputBox.value;
        const cursorPos = inputBox.selectionStart;
        // Find the current token (after last comma before cursor)
        const beforeCursor = value.slice(0, cursorPos);
        const lastComma = beforeCursor.lastIndexOf(',');
        const token = beforeCursor.slice(lastComma + 1).trim();
        // If token is empty, hide dropdown
        if (!token) {
            dropdown.style.display = 'none';
            return;
        }
        // If token looks like attribute (no operator yet), suggest attributes
        const opMatch = token.match(/(=|!=|>=|<=|>|<)/);
        if (!opMatch) {
            // Suggest attributes
            const suggestions = Object.keys(this.attributes).filter(attr => attr.toLowerCase().startsWith(token.toLowerCase()));
            if (suggestions.length === 0) {
                dropdown.style.display = 'none';
                return;
            }
            dropdown.innerHTML = suggestions.map((attr, i) => `<div class="autocomplete-item" data-index="${i}">${attr}</div>`).join('');
            this.autocompleteActiveIndex = 0;
            this.updateAutocompleteActive(dropdown);
            dropdown.style.display = 'block';
            // Click handler
            Array.from(dropdown.children).forEach((item, i) => {
                item.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    this.insertAutocomplete(inputBox, attr, lastComma + 1, cursorPos);
                    dropdown.style.display = 'none';
                });
            });
            return;
        }
        // If token has attribute and operator, suggest operators if just after attribute
        const attrMatch = token.match(/^([a-zA-Z0-9_ ]+)\s*$/);
        if (attrMatch && token.length === attrMatch[1].length) {
            // Suggest operators
            const operators = ['=', '!=', '>', '<', '>=', '<='];
            dropdown.innerHTML = operators.map((op, i) => `<div class="autocomplete-item" data-index="${i}">${op}</div>`).join('');
            this.autocompleteActiveIndex = 0;
            this.updateAutocompleteActive(dropdown);
            dropdown.style.display = 'block';
            Array.from(dropdown.children).forEach((item, i) => {
                item.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    this.insertAutocomplete(inputBox, op, lastComma + 1 + attrMatch[1].length, cursorPos);
                    dropdown.style.display = 'none';
                });
            });
            return;
        }
        dropdown.style.display = 'none';
    }

    insertAutocomplete(inputBox, text, start, end) {
        const value = inputBox.value;
        inputBox.value = value.slice(0, start) + text + value.slice(end);
        // Move cursor to after inserted text
        inputBox.selectionStart = inputBox.selectionEnd = start + text.length;
        inputBox.focus();
    }

    moveAutocompleteSelection(delta, dropdown) {
        const items = Array.from(dropdown.children);
        if (!items.length) return;
        if (this.autocompleteActiveIndex === undefined) this.autocompleteActiveIndex = 0;
        this.autocompleteActiveIndex = (this.autocompleteActiveIndex + delta + items.length) % items.length;
        this.updateAutocompleteActive(dropdown);
    }

    updateAutocompleteActive(dropdown) {
        const items = Array.from(dropdown.children);
        items.forEach((item, i) => {
            if (i === this.autocompleteActiveIndex) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    selectAutocomplete(dropdown, inputBox) {
        const items = Array.from(dropdown.children);
        if (this.autocompleteActiveIndex !== undefined && items[this.autocompleteActiveIndex]) {
            items[this.autocompleteActiveIndex].dispatchEvent(new Event('mousedown'));
        }
    }
}

// Global instance for event handling
let queryBuilder;

document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('smartQueryBuilder');
    if (container) {
        queryBuilder = new SmartQueryBuilder('smartQueryBuilder');
    }
});

// Integration with consumer tab: load attributes when dataset is selected, handle query execution
// This assumes the consumer tab JS exposes a way to get the current dataset and its attributes
window.addEventListener('datasetSelected', function(e) {
    if (queryBuilder && e.detail) {
        console.log('Smart Query Builder: Dataset selected', e.detail);
        queryBuilder.setDataset(e.detail.dataset, e.detail.attributes);
        
        // Show the Smart Query Builder section when a dataset is selected
        const smartQuerySection = document.getElementById('smartQueryBuilderSection');
        if (smartQuerySection) {
            smartQuerySection.classList.remove('hidden');
        }
    }
});

// Handle query execution and integrate with existing consumer results
document.getElementById('smartQueryBuilder').addEventListener('queryExecute', function(e) {
    console.log('Smart Query Builder: Executing query', e.detail);
    
    // Use the same results section as the classic consumer flow
    // e.detail contains: selected_attributes, filters, num_values_k, dp_epsilon
    
    // Show loading state
    const resultsSection = document.getElementById('results');
    const dataTable = document.getElementById('dataTable');
    const privacyNote = document.getElementById('privacyNote');
    
    resultsSection.classList.remove('hidden');
    privacyNote.classList.remove('hidden');
    privacyNote.textContent = 'Processing query with differential privacy...';
    dataTable.innerHTML = '<div class="loading">Processing your query...</div>';
    
    // Send query to backend
    fetch(`${window.API_BASE_URL || 'http://localhost:5001'}/marketplace/datasets/${window.selectedDatasetId}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            selected_attributes: e.detail.selected_attributes,
            num_values_k: e.detail.num_values_k,
            dp_epsilon: e.detail.dp_epsilon,
            filters: e.detail.filters
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log('Query result:', data);
        
        // Display results in the #results section
        if (data && data.data) {
            privacyNote.textContent = data.privacy_note || 'Query completed with differential privacy applied.';
            
            // Render data table
            const table = document.createElement('table');
            table.className = 'data-table';
            
            if (data.data.length > 0) {
                const headers = Object.keys(data.data[0]);
                const thead = document.createElement('thead');
                const headerRow = document.createElement('tr');
                headers.forEach(header => {
                    const th = document.createElement('th');
                    th.textContent = header;
                    headerRow.appendChild(th);
                });
                thead.appendChild(headerRow);
                table.appendChild(thead);
                
                const tbody = document.createElement('tbody');
                data.data.forEach(row => {
                    const tr = document.createElement('tr');
                    headers.forEach(header => {
                        const td = document.createElement('td');
                        td.textContent = row[header] || '';
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);
            } else {
                const tr = document.createElement('tr');
                const td = document.createElement('td');
                td.colSpan = 10;
                td.textContent = 'No data returned.';
                tr.appendChild(td);
                table.appendChild(tr);
            }
            
            dataTable.innerHTML = '';
            dataTable.appendChild(table);
        } else {
            privacyNote.textContent = data.error || 'No data returned.';
            dataTable.innerHTML = '<p>No data available for the specified query.</p>';
        }
    })
    .catch(err => {
        console.error('Query execution error:', err);
        privacyNote.textContent = 'Error executing query. Please try again.';
        dataTable.innerHTML = '<p>An error occurred while processing your query.</p>';
    });
});
