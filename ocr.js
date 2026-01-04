/**
 * Neural OCR Demo
 * A neural network digit recognition application
 */

var ocrDemo = {
    // Configuration
    CANVAS_WIDTH: 280,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 14,
    HOST: 'http://localhost',
    PORT: '8000',
    BATCH_SIZE: 1,

    // State
    data: [],
    trainArray: [],
    trainingRequestCount: 0,
    isDrawing: false,
    hasDrawn: false,
    lastPredictionData: null,  // Store last input for feedback training
    lastPredictionLabel: null, // Store last prediction for confirmation

    // ============================================
    // INITIALIZATION
    // ============================================

    onLoadFunction: function() {
        this.resetCanvas();
        this.setupEventListeners();
        this.initializeVisualization();
        this.loadAndDisplayWeights();
        this.setupKeyboardShortcuts();
    },

    setupEventListeners: function() {
        var canvas = document.getElementById('canvas');
        if (!canvas) return;

        var self = this;

        // Mouse events
        canvas.addEventListener('mousedown', function(e) {
            self.startDrawing(e);
        });
        canvas.addEventListener('mousemove', function(e) {
            self.draw(e);
        });
        canvas.addEventListener('mouseup', function() {
            self.stopDrawing();
        });
        canvas.addEventListener('mouseleave', function() {
            self.stopDrawing();
        });

        // Touch events for mobile
        canvas.addEventListener('touchstart', function(e) {
            e.preventDefault();
            self.startDrawing(e.touches[0]);
        }, { passive: false });
        canvas.addEventListener('touchmove', function(e) {
            e.preventDefault();
            self.draw(e.touches[0]);
        }, { passive: false });
        canvas.addEventListener('touchend', function() {
            self.stopDrawing();
        });

        // Input field - only allow digits
        var digitInput = document.getElementById('digit');
        if (digitInput) {
            digitInput.addEventListener('input', function(e) {
                this.value = this.value.replace(/[^0-9]/g, '').slice(0, 1);
            });
            digitInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    self.train();
                }
            });
        }
    },

    setupKeyboardShortcuts: function() {
        var self = this;
        document.addEventListener('keydown', function(e) {
            // Don't trigger shortcuts when typing in input
            if (e.target.tagName === 'INPUT') return;

            switch(e.key.toLowerCase()) {
                case 't':
                    self.test();
                    break;
                case 'c':
                case 'escape':
                    self.resetCanvas();
                    break;
                case 'r':
                    if (e.ctrlKey || e.metaKey) return; // Don't override browser refresh
                    self.test();
                    break;
            }
        });
    },

    // ============================================
    // CANVAS OPERATIONS
    // ============================================

    resetCanvas: function() {
        var canvas = document.getElementById('canvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);

        // Initialize data array
        this.data = [];
        for (var i = 0; i < this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH; i++) {
            this.data.push(0);
        }

        this.drawGrid(ctx);
        this.hasDrawn = false;
        this.updateCanvasOverlay();
        this.updatePredictionDisplay(null, null);
        this.hideFeedbackPanel();

        // Stop particle animation when canvas is cleared
        this.stopParticleAnimation();
    },

    drawGrid: function(ctx) {
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.15)';
        ctx.lineWidth = 1;

        for (var i = 1; i < this.TRANSLATED_WIDTH; i++) {
            var pos = i * this.PIXEL_WIDTH;

            ctx.beginPath();
            ctx.moveTo(pos, 0);
            ctx.lineTo(pos, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, pos);
            ctx.lineTo(this.CANVAS_WIDTH, pos);
            ctx.stroke();
        }
    },

    startDrawing: function(e) {
        this.isDrawing = true;
        var canvas = document.getElementById('canvas');
        if (canvas) canvas.classList.add('drawing');
        this.draw(e);
    },

    stopDrawing: function() {
        this.isDrawing = false;
        var canvas = document.getElementById('canvas');
        if (canvas) canvas.classList.remove('drawing');
    },

    draw: function(e) {
        if (!this.isDrawing) return;

        var canvas = document.getElementById('canvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        var rect = canvas.getBoundingClientRect();
        var scaleX = canvas.width / rect.width;
        var scaleY = canvas.height / rect.height;

        var x = (e.clientX - rect.left) * scaleX;
        var y = (e.clientY - rect.top) * scaleY;

        this.fillSquare(ctx, x, y);

        if (!this.hasDrawn) {
            this.hasDrawn = true;
            this.updateCanvasOverlay();
        }
    },

    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);

        if (xPixel < 0 || yPixel < 0 || xPixel >= this.TRANSLATED_WIDTH || yPixel >= this.TRANSLATED_WIDTH) {
            return;
        }

        var index = yPixel * this.TRANSLATED_WIDTH + xPixel;
        this.data[index] = 1;

        // Draw with a slight glow effect
        var px = xPixel * this.PIXEL_WIDTH;
        var py = yPixel * this.PIXEL_WIDTH;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(px + 1, py + 1, this.PIXEL_WIDTH - 2, this.PIXEL_WIDTH - 2);
    },

    updateCanvasOverlay: function() {
        var overlay = document.getElementById('canvas-overlay');
        if (overlay) {
            overlay.classList.toggle('hidden', this.hasDrawn);
        }
    },

    // ============================================
    // TRAINING AND PREDICTION
    // ============================================

    train: function() {
        var digitVal = document.getElementById('digit').value;

        if (!digitVal) {
            this.showToast('warning', 'Missing Label', 'Please enter the digit value (0-9) to train the network.');
            document.getElementById('digit').focus();
            return;
        }

        if (this.data.indexOf(1) < 0) {
            this.showToast('warning', 'Empty Canvas', 'Please draw a digit before training.');
            return;
        }

        var label = parseInt(digitVal);
        if (isNaN(label) || label < 0 || label > 9) {
            this.showToast('error', 'Invalid Label', 'Please enter a digit between 0 and 9.');
            return;
        }

        // Center the drawing before training for consistency
        var processedData = this.preprocessInput(this.data.slice());
        this.trainArray.push({ "y0": processedData, "label": label });
        this.trainingRequestCount++;

        if (this.trainingRequestCount >= this.BATCH_SIZE) {
            this.showLoading('Training network...');

            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        } else {
            this.showToast('info', 'Sample Added', 'Training sample added. Draw more samples or click Train again.');
        }

        // Reset for next input
        this.resetCanvas();
        document.getElementById('digit').value = '';
        document.getElementById('digit').focus();
    },

    test: function() {
        if (this.data.indexOf(1) < 0) {
            this.showToast('warning', 'Empty Canvas', 'Please draw a digit to recognize.');
            return;
        }

        this.showLoading('Recognizing...');
        this.updateStatusBadge('Processing...');

        // Center and normalize the drawing before sending
        var processedData = this.preprocessInput(this.data.slice());

        // Store for potential feedback training
        this.lastPredictionData = processedData;

        var json = {
            image: processedData,
            predict: true
        };

        this.sendData(json);
    },

    // ============================================
    // INPUT PREPROCESSING - Center and normalize
    // ============================================

    preprocessInput: function(data) {
        var size = this.TRANSLATED_WIDTH; // 20

        // Find bounding box of the drawing
        var minX = size, maxX = 0, minY = size, maxY = 0;
        var hasPixels = false;

        for (var y = 0; y < size; y++) {
            for (var x = 0; x < size; x++) {
                if (data[y * size + x] > 0) {
                    hasPixels = true;
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        if (!hasPixels) return data;

        // Calculate dimensions of the drawing
        var drawingWidth = maxX - minX + 1;
        var drawingHeight = maxY - minY + 1;

        // Calculate center of mass for more accurate centering
        var centerX = 0, centerY = 0, totalMass = 0;
        for (var y = 0; y < size; y++) {
            for (var x = 0; x < size; x++) {
                var val = data[y * size + x];
                if (val > 0) {
                    centerX += x * val;
                    centerY += y * val;
                    totalMass += val;
                }
            }
        }
        centerX = Math.round(centerX / totalMass);
        centerY = Math.round(centerY / totalMass);

        // Calculate offset to center the drawing
        var targetCenterX = Math.floor(size / 2);
        var targetCenterY = Math.floor(size / 2);
        var offsetX = targetCenterX - centerX;
        var offsetY = targetCenterY - centerY;

        // Create new centered array
        var centered = [];
        for (var i = 0; i < size * size; i++) {
            centered.push(0);
        }

        // Copy pixels with offset
        for (var y = 0; y < size; y++) {
            for (var x = 0; x < size; x++) {
                var srcVal = data[y * size + x];
                if (srcVal > 0) {
                    var newX = x + offsetX;
                    var newY = y + offsetY;
                    // Only copy if within bounds
                    if (newX >= 0 && newX < size && newY >= 0 && newY < size) {
                        centered[newY * size + newX] = srcVal;
                    }
                }
            }
        }

        return centered;
    },

    // ============================================
    // NETWORK COMMUNICATION
    // ============================================

    sendData: function(json) {
        var self = this;
        var xmlHttp = new XMLHttpRequest();
        var url = this.HOST + ':' + this.PORT + '/';

        xmlHttp.open('POST', url, true);
        xmlHttp.setRequestHeader('Content-Type', 'application/json');

        xmlHttp.onload = function() {
            self.hideLoading();
            self.receiveResponse(xmlHttp);
        };

        xmlHttp.onerror = function() {
            self.hideLoading();
            self.showToast('error', 'Connection Failed', 'Could not connect to the server. Make sure the Python server is running.');
            self.updateStatusBadge('Error');
        };

        xmlHttp.ontimeout = function() {
            self.hideLoading();
            self.showToast('error', 'Timeout', 'Request timed out. Please try again.');
            self.updateStatusBadge('Timeout');
        };

        xmlHttp.timeout = 30000;
        xmlHttp.send(JSON.stringify(json));
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status !== 200) {
            this.showToast('error', 'Server Error', 'Server returned status ' + xmlHttp.status);
            this.updateStatusBadge('Error');
            return;
        }

        try {
            var response = JSON.parse(xmlHttp.responseText);

            if (response.type === 'test') {
                this.handlePredictionResponse(response);
            } else if (response.type === 'train') {
                this.handleTrainResponse(response);
            } else if (response.type === 'weights') {
                this.handleWeightsResponse(response);
            }
        } catch (e) {
            this.showToast('error', 'Parse Error', 'Could not parse server response.');
        }
    },

    handlePredictionResponse: function(response) {
        var prediction = response.result;
        var activations = response.activations;

        // Store prediction for feedback
        this.lastPredictionLabel = prediction;

        if (activations) {
            this.vizState.activations = activations;
            this.vizState.lastPrediction = prediction;

            // Update all visualizations - input shows the centered/preprocessed version
            this.renderInputHeatmap(activations.input);
            this.renderConfidenceChart(activations.output, prediction);
            this.renderNetworkArchitecture();

            // Update prediction display
            var confidence = activations.output[prediction];
            this.updatePredictionDisplay(prediction, confidence);
        }

        this.updateStatusBadge('Ready');

        // Show feedback panel
        this.showFeedbackPanel();

        // Show confidence level in toast
        var confPercent = activations ? (activations.output[prediction] * 100).toFixed(0) : 0;
        this.showToast('success', 'Prediction: ' + prediction, confPercent + '% confidence');
    },

    handleTrainResponse: function(response) {
        if (response.weights) {
            this.vizState.weights = response.weights;
            this.renderNetworkArchitecture();
        }

        this.showToast('success', 'Training Complete', 'Network weights have been updated.');
    },

    handleWeightsResponse: function(response) {
        this.vizConfig.hiddenNodes = response.hiddenNodes;
        this.vizState.weights = {
            theta1: response.theta1,
            theta2: response.theta2
        };

        // Update footer with hidden node count
        var hiddenCountEl = document.getElementById('hidden-count');
        if (hiddenCountEl) {
            hiddenCountEl.textContent = response.hiddenNodes;
        }

        this.renderNetworkArchitecture();
    },

    // ============================================
    // UI HELPERS
    // ============================================

    updatePredictionDisplay: function(prediction, confidence) {
        var numberEl = document.getElementById('prediction-result');
        var confEl = document.getElementById('prediction-confidence');

        if (prediction === null) {
            numberEl.textContent = '?';
            numberEl.classList.add('uncertain');
            confEl.textContent = 'Draw a digit to see prediction';
        } else {
            numberEl.textContent = prediction;
            numberEl.classList.remove('uncertain');
            confEl.textContent = (confidence * 100).toFixed(1) + '% confidence';
        }
    },

    updateStatusBadge: function(text) {
        var badge = document.getElementById('status-badge');
        if (badge) badge.textContent = text;
    },

    // ============================================
    // FEEDBACK SYSTEM
    // ============================================

    showFeedbackPanel: function() {
        var panel = document.getElementById('feedback-panel');
        var correctionInput = document.getElementById('correction-input');
        if (panel) {
            panel.classList.remove('hidden');
        }
        if (correctionInput) {
            correctionInput.classList.add('hidden');
        }
    },

    hideFeedbackPanel: function() {
        var panel = document.getElementById('feedback-panel');
        var correctionInput = document.getElementById('correction-input');
        if (panel) {
            panel.classList.add('hidden');
        }
        if (correctionInput) {
            correctionInput.classList.add('hidden');
        }
    },

    showCorrectionInput: function() {
        var correctionInput = document.getElementById('correction-input');
        if (correctionInput) {
            correctionInput.classList.remove('hidden');
        }
    },

    confirmPrediction: function() {
        if (!this.lastPredictionData || this.lastPredictionLabel === null) {
            this.showToast('warning', 'No Prediction', 'Nothing to confirm.');
            return;
        }

        // Train with the predicted label
        this.trainWithFeedback(this.lastPredictionLabel);
        this.showToast('success', 'Thanks!', 'Model reinforced with label: ' + this.lastPredictionLabel);
        this.hideFeedbackPanel();
        this.resetCanvas();
    },

    correctPrediction: function(correctLabel) {
        if (!this.lastPredictionData) {
            this.showToast('warning', 'No Prediction', 'Nothing to correct.');
            return;
        }

        // Train with the correct label
        this.trainWithFeedback(correctLabel);
        this.showToast('info', 'Corrected!', 'Model updated: ' + this.lastPredictionLabel + ' → ' + correctLabel);
        this.hideFeedbackPanel();
        this.resetCanvas();
    },

    trainWithFeedback: function(label) {
        if (!this.lastPredictionData) return;

        this.showLoading('Learning...');

        var json = {
            trainArray: [{ "y0": this.lastPredictionData, "label": label }],
            train: true
        };

        this.sendData(json);

        // Clear stored prediction data
        this.lastPredictionData = null;
        this.lastPredictionLabel = null;
    },

    showLoading: function(text) {
        var overlay = document.getElementById('loading-overlay');
        var loadingText = document.getElementById('loading-text');
        if (overlay) overlay.classList.remove('hidden');
        if (loadingText) loadingText.textContent = text || 'Processing...';
    },

    hideLoading: function() {
        var overlay = document.getElementById('loading-overlay');
        if (overlay) overlay.classList.add('hidden');
    },

    // ============================================
    // TOAST NOTIFICATIONS
    // ============================================

    showToast: function(type, title, message) {
        var container = document.getElementById('toast-container');
        if (!container) return;

        var icons = {
            success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
            warning: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
        };

        var toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.innerHTML =
            '<div class="toast-icon">' + icons[type] + '</div>' +
            '<div class="toast-content">' +
                '<div class="toast-title">' + title + '</div>' +
                '<div class="toast-message">' + message + '</div>' +
            '</div>' +
            '<button class="toast-close" aria-label="Close">' +
                '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
                    '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>' +
                '</svg>' +
            '</button>';

        var closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', function() {
            removeToast(toast);
        });

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(function() {
            removeToast(toast);
        }, 5000);

        function removeToast(t) {
            if (!t.parentNode) return;
            t.classList.add('removing');
            setTimeout(function() {
                if (t.parentNode) t.parentNode.removeChild(t);
            }, 300);
        }
    },

    // ============================================
    // VISUALIZATION SYSTEM
    // ============================================

    initializeVisualization: function() {
        this.vizConfig = {
            width: 1200,
            height: 300,
            inputNodes: 400,
            hiddenNodes: 15,
            outputNodes: 10,
            inputSample: 25, // 5x5 grid for more detail
            nodeRadius: 10,
            particleSpeed: 2,
            glowIntensity: 0.8
        };

        this.vizState = {
            weights: null,
            activations: null,
            lastPrediction: null,
            particles: [],
            animationFrame: null,
            isAnimating: false
        };

        this.initInputHeatmap();
        this.initNetworkCanvas();
    },

    initNetworkCanvas: function() {
        // Replace SVG with canvas for better animation performance
        var container = document.querySelector('.network-viz-wrapper');
        if (!container) return;

        // Create canvas element
        var canvas = document.createElement('canvas');
        canvas.id = 'network-canvas';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.borderRadius = '12px';

        // Replace SVG with canvas
        var svg = document.getElementById('network-viz');
        if (svg) {
            svg.style.display = 'none';
        }
        container.appendChild(canvas);

        // Handle resize
        var self = this;
        window.addEventListener('resize', function() {
            self.resizeNetworkCanvas();
            if (self.vizState.weights) {
                self.renderNetworkArchitecture();
            }
        });

        this.resizeNetworkCanvas();
    },

    resizeNetworkCanvas: function() {
        var canvas = document.getElementById('network-canvas');
        if (!canvas) return;

        var rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';

        var ctx = canvas.getContext('2d');
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    },

    initInputHeatmap: function() {
        var canvas = document.getElementById('input-heatmap');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, 200, 200);
    },

    loadAndDisplayWeights: function() {
        var self = this;
        var xmlHttp = new XMLHttpRequest();
        var url = this.HOST + ':' + this.PORT + '/';

        xmlHttp.open('POST', url, true);
        xmlHttp.setRequestHeader('Content-Type', 'application/json');

        xmlHttp.onload = function() {
            if (xmlHttp.status === 200) {
                try {
                    var response = JSON.parse(xmlHttp.responseText);
                    if (response.type === 'weights') {
                        self.handleWeightsResponse(response);
                    }
                } catch (e) {
                    // Silently fail - weights will be loaded on first interaction
                }
            }
        };

        xmlHttp.onerror = function() {
            self.showToast('warning', 'Server Offline', 'Could not connect to the neural network server. Please start the Python server.');
        };

        xmlHttp.send(JSON.stringify({ getWeights: true }));
    },

    // ============================================
    // VISUALIZATION RENDERING
    // ============================================

    renderInputHeatmap: function(inputVector) {
        var canvas = document.getElementById('input-heatmap');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        var cellSize = 10;

        for (var i = 0; i < 20; i++) {
            for (var j = 0; j < 20; j++) {
                var idx = i * 20 + j;
                var value = inputVector[idx];

                // Create a gradient from dark to cyan/white
                var intensity = value;
                var r = Math.floor(30 + intensity * 180);
                var g = Math.floor(30 + intensity * 225);
                var b = Math.floor(50 + intensity * 205);

                ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

                // Add glow effect for active pixels
                if (value > 0.5) {
                    ctx.shadowColor = 'rgba(99, 200, 255, ' + (value * 0.5) + ')';
                    ctx.shadowBlur = 8;
                    ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                    ctx.shadowBlur = 0;
                }
            }
        }
    },

    renderConfidenceChart: function(outputActivations, prediction) {
        var svg = document.getElementById('confidence-chart');
        if (!svg) return;

        svg.innerHTML = '';

        var width = svg.clientWidth || 300;
        var height = 180;
        var barHeight = 14;
        var barSpacing = 4;
        var maxBarWidth = width - 80;
        var startX = 30;
        var startY = 5;

        // Create gradient definitions
        var defs = this.createSVGElement('defs', {});

        var gradientSuccess = this.createSVGElement('linearGradient', {
            id: 'barGradientSuccess',
            x1: '0%', y1: '0%', x2: '100%', y2: '0%'
        });
        gradientSuccess.appendChild(this.createSVGElement('stop', { offset: '0%', 'stop-color': '#10b981' }));
        gradientSuccess.appendChild(this.createSVGElement('stop', { offset: '100%', 'stop-color': '#34d399' }));
        defs.appendChild(gradientSuccess);

        var gradientNormal = this.createSVGElement('linearGradient', {
            id: 'barGradientNormal',
            x1: '0%', y1: '0%', x2: '100%', y2: '0%'
        });
        gradientNormal.appendChild(this.createSVGElement('stop', { offset: '0%', 'stop-color': '#6366f1' }));
        gradientNormal.appendChild(this.createSVGElement('stop', { offset: '100%', 'stop-color': '#818cf8' }));
        defs.appendChild(gradientNormal);

        svg.appendChild(defs);

        for (var i = 0; i < 10; i++) {
            var confidence = outputActivations[i];
            var barWidth = Math.max(2, confidence * maxBarWidth);
            var y = startY + i * (barHeight + barSpacing);
            var isPrediction = i === prediction;

            // Background bar
            var bgRect = this.createSVGElement('rect', {
                x: startX,
                y: y,
                width: maxBarWidth,
                height: barHeight,
                fill: '#1a1a2e',
                rx: 4
            });
            svg.appendChild(bgRect);

            // Confidence bar with gradient
            var rect = this.createSVGElement('rect', {
                x: startX,
                y: y,
                width: barWidth,
                height: barHeight,
                fill: isPrediction ? 'url(#barGradientSuccess)' : 'url(#barGradientNormal)',
                rx: 4,
                opacity: isPrediction ? 1 : 0.7
            });
            svg.appendChild(rect);

            // Glow effect for prediction
            if (isPrediction && confidence > 0.5) {
                var glow = this.createSVGElement('rect', {
                    x: startX,
                    y: y,
                    width: barWidth,
                    height: barHeight,
                    fill: 'none',
                    stroke: '#10b981',
                    'stroke-width': 2,
                    rx: 4,
                    filter: 'url(#glow)'
                });
                svg.appendChild(glow);
            }

            // Digit label
            var label = this.createSVGElement('text', {
                x: startX - 10,
                y: y + barHeight / 2 + 4,
                'text-anchor': 'end',
                'font-size': '12',
                'font-weight': isPrediction ? 'bold' : 'normal',
                fill: isPrediction ? '#10b981' : '#a0a0b0'
            });
            label.textContent = i;
            svg.appendChild(label);

            // Confidence percentage
            var valueText = this.createSVGElement('text', {
                x: startX + maxBarWidth + 8,
                y: y + barHeight / 2 + 4,
                'font-size': '11',
                fill: isPrediction ? '#10b981' : '#6b7280'
            });
            valueText.textContent = (confidence * 100).toFixed(1) + '%';
            svg.appendChild(valueText);
        }
    },

    renderNetworkArchitecture: function() {
        var canvas = document.getElementById('network-canvas');
        if (!canvas || !this.vizState.weights) return;

        var ctx = canvas.getContext('2d');
        var rect = canvas.parentElement.getBoundingClientRect();
        var width = rect.width;
        var height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw background gradient
        var bgGradient = ctx.createLinearGradient(0, 0, 0, height);
        bgGradient.addColorStop(0, 'rgba(15, 15, 26, 0.95)');
        bgGradient.addColorStop(0.5, 'rgba(22, 33, 62, 0.9)');
        bgGradient.addColorStop(1, 'rgba(15, 15, 26, 0.95)');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);

        // Draw grid pattern
        this.drawBackgroundGrid(ctx, width, height);

        var weights = this.vizState.weights;
        var theta1 = weights.theta1;
        var theta2 = weights.theta2;

        // Calculate layer positions
        var inputX = 100;
        var hiddenX = width / 2;
        var outputX = width - 100;

        // Sample input indices (5x5 grid from 20x20)
        var inputSampleIndices = [];
        for (var i = 0; i < 5; i++) {
            for (var j = 0; j < 5; j++) {
                inputSampleIndices.push((i * 4) * 20 + (j * 4));
            }
        }

        var inputPositions = this.calculateNodePositions(25, height, inputX);
        var hiddenPositions = this.calculateNodePositions(this.vizConfig.hiddenNodes, height, hiddenX);
        var outputPositions = this.calculateNodePositions(10, height, outputX);

        // Store positions for animation
        this.vizState.nodePositions = {
            input: inputPositions,
            hidden: hiddenPositions,
            output: outputPositions,
            inputIndices: inputSampleIndices
        };

        // Get activations
        var inputActivations = this.vizState.activations ? this.vizState.activations.input : null;
        var hiddenActivations = this.vizState.activations ? this.vizState.activations.hidden : null;
        var outputActivations = this.vizState.activations ? this.vizState.activations.output : null;

        // Draw connections with glow
        this.drawCanvasConnections(ctx, inputPositions, hiddenPositions, theta1, inputSampleIndices, true, inputActivations, hiddenActivations);
        this.drawCanvasConnections(ctx, hiddenPositions, outputPositions, theta2, null, false, hiddenActivations, outputActivations);

        // Draw nodes with glow effects
        this.drawCanvasNodes(ctx, inputPositions, { r: 99, g: 102, b: 241 }, inputActivations, inputSampleIndices);
        this.drawCanvasNodes(ctx, hiddenPositions, { r: 139, g: 92, b: 246 }, hiddenActivations, null);
        this.drawCanvasNodes(ctx, outputPositions, { r: 245, g: 158, b: 11 }, outputActivations, null, true);

        // Draw layer labels
        this.drawCanvasLabel(ctx, inputX, 25, 'Input (5x5)', 'rgba(99, 102, 241, 0.9)');
        this.drawCanvasLabel(ctx, hiddenX, 25, 'Hidden (' + this.vizConfig.hiddenNodes + ')', 'rgba(139, 92, 246, 0.9)');
        this.drawCanvasLabel(ctx, outputX, 25, 'Output (0-9)', 'rgba(245, 158, 11, 0.9)');

        // Start particle animation if we have activations
        if (this.vizState.activations && !this.vizState.isAnimating) {
            this.startParticleAnimation();
        }
    },

    drawBackgroundGrid: function(ctx, width, height) {
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.05)';
        ctx.lineWidth = 1;

        var spacing = 30;
        for (var x = 0; x < width; x += spacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (var y = 0; y < height; y += spacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    },

    calculateNodePositions: function(numNodes, height, x) {
        var positions = [];
        var usableHeight = height - 70;
        var spacing = usableHeight / (numNodes + 1);

        for (var i = 0; i < numNodes; i++) {
            positions.push({ x: x, y: 45 + spacing * (i + 1) });
        }
        return positions;
    },

    drawCanvasConnections: function(ctx, fromPositions, toPositions, weights, sampleIndices, isInputLayer, fromActivations, toActivations) {
        var maxWeight = 0.001;
        for (var i = 0; i < weights.length; i++) {
            for (var j = 0; j < weights[i].length; j++) {
                maxWeight = Math.max(maxWeight, Math.abs(weights[i][j]));
            }
        }

        var threshold = 0.15;

        for (var i = 0; i < toPositions.length; i++) {
            for (var j = 0; j < fromPositions.length; j++) {
                var actualFromIdx = isInputLayer && sampleIndices ? sampleIndices[j] : j;
                var weight = weights[i][actualFromIdx];
                var normalizedWeight = weight / maxWeight;

                if (Math.abs(normalizedWeight) > threshold) {
                    // Get activation-based intensity
                    var fromAct = fromActivations ? fromActivations[actualFromIdx] || 0 : 0.3;
                    var toAct = toActivations ? toActivations[i] || 0 : 0.3;
                    var actIntensity = (fromAct + toAct) / 2;

                    var isPositive = weight > 0;
                    var baseAlpha = Math.abs(normalizedWeight) * 0.4;
                    var alpha = baseAlpha + actIntensity * 0.3;

                    // Create gradient along the connection
                    var gradient = ctx.createLinearGradient(
                        fromPositions[j].x, fromPositions[j].y,
                        toPositions[i].x, toPositions[i].y
                    );

                    if (isPositive) {
                        gradient.addColorStop(0, 'rgba(34, 197, 94, ' + (alpha * 0.5) + ')');
                        gradient.addColorStop(0.5, 'rgba(74, 222, 128, ' + alpha + ')');
                        gradient.addColorStop(1, 'rgba(34, 197, 94, ' + (alpha * 0.5) + ')');
                    } else {
                        gradient.addColorStop(0, 'rgba(239, 68, 68, ' + (alpha * 0.5) + ')');
                        gradient.addColorStop(0.5, 'rgba(248, 113, 113, ' + alpha + ')');
                        gradient.addColorStop(1, 'rgba(239, 68, 68, ' + (alpha * 0.5) + ')');
                    }

                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = Math.abs(normalizedWeight) * 2 + 0.5;

                    // Add glow for active connections
                    if (actIntensity > 0.5) {
                        ctx.shadowColor = isPositive ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)';
                        ctx.shadowBlur = 8;
                    }

                    ctx.beginPath();
                    ctx.moveTo(fromPositions[j].x, fromPositions[j].y);
                    ctx.lineTo(toPositions[i].x, toPositions[i].y);
                    ctx.stroke();

                    ctx.shadowBlur = 0;
                }
            }
        }
    },

    drawCanvasNodes: function(ctx, positions, baseColor, activations, sampleIndices, showLabels) {
        for (var i = 0; i < positions.length; i++) {
            var activation = 0.3;
            if (activations) {
                var actualIdx = sampleIndices ? sampleIndices[i] : i;
                activation = activations[actualIdx] || 0.3;
            }

            var radius = this.vizConfig.nodeRadius;
            var x = positions[i].x;
            var y = positions[i].y;

            // Outer glow
            var glowRadius = radius + 8 + activation * 6;
            var glowGradient = ctx.createRadialGradient(x, y, radius, x, y, glowRadius);
            glowGradient.addColorStop(0, 'rgba(' + baseColor.r + ',' + baseColor.g + ',' + baseColor.b + ',' + (activation * 0.4) + ')');
            glowGradient.addColorStop(1, 'rgba(' + baseColor.r + ',' + baseColor.g + ',' + baseColor.b + ', 0)');

            ctx.fillStyle = glowGradient;
            ctx.beginPath();
            ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
            ctx.fill();

            // Node gradient
            var nodeGradient = ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, 0, x, y, radius);
            var intensity = 0.4 + activation * 0.6;
            nodeGradient.addColorStop(0, 'rgba(' + Math.min(255, baseColor.r + 60) + ',' + Math.min(255, baseColor.g + 60) + ',' + Math.min(255, baseColor.b + 60) + ',' + intensity + ')');
            nodeGradient.addColorStop(1, 'rgba(' + baseColor.r + ',' + baseColor.g + ',' + baseColor.b + ',' + intensity + ')');

            // Draw node
            ctx.fillStyle = nodeGradient;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();

            // Node border
            ctx.strokeStyle = 'rgba(' + baseColor.r + ',' + baseColor.g + ',' + baseColor.b + ', 0.8)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Highlight ring for high activation
            if (activation > 0.7) {
                ctx.strokeStyle = 'rgba(255, 255, 255, ' + ((activation - 0.7) * 2) + ')';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Labels for output layer
            if (showLabels) {
                ctx.font = '11px Inter, sans-serif';
                ctx.fillStyle = activation > 0.5 ? 'rgba(255, 255, 255, 0.9)' : 'rgba(160, 160, 176, 0.8)';
                ctx.textAlign = 'left';
                ctx.fillText(i, x + radius + 10, y + 4);
            }
        }
    },

    drawCanvasLabel: function(ctx, x, y, text, color) {
        ctx.font = '600 12px Inter, sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        ctx.fillText(text, x, y);
    },

    // ============================================
    // PARTICLE ANIMATION SYSTEM
    // ============================================

    startParticleAnimation: function() {
        // Stop any existing animation first
        this.stopParticleAnimation();

        this.vizState.isAnimating = true;
        this.vizState.particles = [];
        this.vizState.particleSpawnTimer = 0;
        this.createInitialParticles();
        this.animateParticles();
    },

    stopParticleAnimation: function() {
        this.vizState.isAnimating = false;
        if (this.vizState.animationFrame) {
            cancelAnimationFrame(this.vizState.animationFrame);
            this.vizState.animationFrame = null;
        }
    },

    createInitialParticles: function() {
        if (!this.vizState.nodePositions || !this.vizState.activations) return;

        // Create a good number of initial particles spread across progress
        for (var i = 0; i < 30; i++) {
            this.spawnParticle(Math.random());
        }
    },

    spawnParticle: function(initialProgress) {
        if (!this.vizState.nodePositions || !this.vizState.activations) return;

        var positions = this.vizState.nodePositions;
        var activations = this.vizState.activations;
        var predIdx = this.vizState.lastPrediction;

        // Randomly choose layer 1 (input->hidden) or layer 2 (hidden->output)
        var layer = Math.random() < 0.6 ? 1 : 2;

        if (layer === 1) {
            // Find active input and hidden nodes
            var activeInputs = [];
            var activeHiddens = [];

            for (var j = 0; j < positions.input.length; j++) {
                var inputIdx = positions.inputIndices[j];
                var inputAct = activations.input[inputIdx] || 0;
                if (inputAct > 0.3) {
                    activeInputs.push({ idx: j, act: inputAct });
                }
            }

            for (var i = 0; i < positions.hidden.length; i++) {
                var hiddenAct = activations.hidden[i] || 0;
                if (hiddenAct > 0.2) {
                    activeHiddens.push({ idx: i, act: hiddenAct });
                }
            }

            if (activeInputs.length > 0 && activeHiddens.length > 0) {
                var fromNode = activeInputs[Math.floor(Math.random() * activeInputs.length)];
                var toNode = activeHiddens[Math.floor(Math.random() * activeHiddens.length)];

                this.vizState.particles.push({
                    fromX: positions.input[fromNode.idx].x,
                    fromY: positions.input[fromNode.idx].y,
                    toX: positions.hidden[toNode.idx].x,
                    toY: positions.hidden[toNode.idx].y,
                    progress: initialProgress || 0,
                    speed: 0.006 + Math.random() * 0.008,
                    size: 2.5 + Math.random() * 2.5,
                    color: { r: 99, g: 180, b: 255 },
                    layer: 1,
                    intensity: (fromNode.act + toNode.act) / 2
                });
            }
        } else {
            // Layer 2: hidden -> output (to prediction)
            var activeHiddens = [];
            for (var i = 0; i < positions.hidden.length; i++) {
                var hiddenAct = activations.hidden[i] || 0;
                if (hiddenAct > 0.3) {
                    activeHiddens.push({ idx: i, act: hiddenAct });
                }
            }

            if (activeHiddens.length > 0 && predIdx !== null && predIdx !== undefined) {
                var fromNode = activeHiddens[Math.floor(Math.random() * activeHiddens.length)];

                this.vizState.particles.push({
                    fromX: positions.hidden[fromNode.idx].x,
                    fromY: positions.hidden[fromNode.idx].y,
                    toX: positions.output[predIdx].x,
                    toY: positions.output[predIdx].y,
                    progress: initialProgress || 0,
                    speed: 0.008 + Math.random() * 0.01,
                    size: 3 + Math.random() * 3,
                    color: { r: 255, g: 200, b: 50 },
                    layer: 2,
                    intensity: fromNode.act
                });
            }
        }
    },

    animateParticles: function() {
        if (!this.vizState.isAnimating) return;

        var canvas = document.getElementById('network-canvas');
        if (!canvas) return;

        var self = this;

        // Redraw the static network (without triggering new animation)
        this.drawNetworkFrame();

        // Draw particles
        var ctx = canvas.getContext('2d');
        var particles = this.vizState.particles;

        // Spawn new particles periodically to maintain density
        this.vizState.particleSpawnTimer = (this.vizState.particleSpawnTimer || 0) + 1;
        if (this.vizState.particleSpawnTimer > 5 && particles.length < 50) {
            this.spawnParticle(0);
            this.vizState.particleSpawnTimer = 0;
        }

        for (var i = particles.length - 1; i >= 0; i--) {
            var p = particles[i];
            p.progress += p.speed;

            // When particle completes journey, respawn it as a new particle
            if (p.progress >= 1) {
                particles.splice(i, 1);
                this.spawnParticle(0);
                continue;
            }

            // Calculate position along path with easing
            var t = p.progress;
            var eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

            var x = p.fromX + (p.toX - p.fromX) * eased;
            var y = p.fromY + (p.toY - p.fromY) * eased;

            // Draw particle trail
            var trailLength = 4;
            for (var tr = 1; tr < trailLength; tr++) {
                var trailT = Math.max(0, t - tr * 0.025);
                var trailEased = trailT < 0.5 ? 2 * trailT * trailT : 1 - Math.pow(-2 * trailT + 2, 2) / 2;
                var trailX = p.fromX + (p.toX - p.fromX) * trailEased;
                var trailY = p.fromY + (p.toY - p.fromY) * trailEased;
                var trailAlpha = (1 - tr / trailLength) * Math.sin(trailT * Math.PI) * 0.5;

                ctx.fillStyle = 'rgba(' + p.color.r + ',' + p.color.g + ',' + p.color.b + ',' + trailAlpha + ')';
                ctx.beginPath();
                ctx.arc(trailX, trailY, p.size * (1 - tr * 0.15), 0, Math.PI * 2);
                ctx.fill();
            }

            // Calculate alpha based on position (fade in at start, fade out at end)
            var alpha = Math.sin(p.progress * Math.PI);
            var intensity = p.intensity || 0.7;

            // Draw outer glow
            ctx.shadowColor = 'rgba(' + p.color.r + ',' + p.color.g + ',' + p.color.b + ', ' + (alpha * 0.8) + ')';
            ctx.shadowBlur = 20;

            // Draw main particle orb
            var gradient = ctx.createRadialGradient(x, y, 0, x, y, p.size);
            gradient.addColorStop(0, 'rgba(255, 255, 255, ' + (alpha * 0.9) + ')');
            gradient.addColorStop(0.3, 'rgba(' + Math.min(255, p.color.r + 50) + ',' + Math.min(255, p.color.g + 50) + ',' + Math.min(255, p.color.b + 50) + ',' + alpha + ')');
            gradient.addColorStop(1, 'rgba(' + p.color.r + ',' + p.color.g + ',' + p.color.b + ',' + (alpha * 0.5) + ')');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, p.size, 0, Math.PI * 2);
            ctx.fill();

            ctx.shadowBlur = 0;
        }

        // Continue animation
        this.vizState.animationFrame = requestAnimationFrame(function() {
            self.animateParticles();
        });
    },

    drawNetworkFrame: function() {
        var canvas = document.getElementById('network-canvas');
        if (!canvas || !this.vizState.weights) return;

        var ctx = canvas.getContext('2d');
        var rect = canvas.parentElement.getBoundingClientRect();
        var width = rect.width;
        var height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw background gradient
        var bgGradient = ctx.createLinearGradient(0, 0, 0, height);
        bgGradient.addColorStop(0, 'rgba(15, 15, 26, 0.95)');
        bgGradient.addColorStop(0.5, 'rgba(22, 33, 62, 0.9)');
        bgGradient.addColorStop(1, 'rgba(15, 15, 26, 0.95)');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, width, height);

        // Draw grid pattern
        this.drawBackgroundGrid(ctx, width, height);

        var positions = this.vizState.nodePositions;
        if (!positions) return;

        var weights = this.vizState.weights;
        var theta1 = weights.theta1;
        var theta2 = weights.theta2;

        // Get activations
        var inputActivations = this.vizState.activations ? this.vizState.activations.input : null;
        var hiddenActivations = this.vizState.activations ? this.vizState.activations.hidden : null;
        var outputActivations = this.vizState.activations ? this.vizState.activations.output : null;

        // Draw connections with glow
        this.drawCanvasConnections(ctx, positions.input, positions.hidden, theta1, positions.inputIndices, true, inputActivations, hiddenActivations);
        this.drawCanvasConnections(ctx, positions.hidden, positions.output, theta2, null, false, hiddenActivations, outputActivations);

        // Draw nodes with glow effects
        this.drawCanvasNodes(ctx, positions.input, { r: 99, g: 102, b: 241 }, inputActivations, positions.inputIndices);
        this.drawCanvasNodes(ctx, positions.hidden, { r: 139, g: 92, b: 246 }, hiddenActivations, null);
        this.drawCanvasNodes(ctx, positions.output, { r: 245, g: 158, b: 11 }, outputActivations, null, true);

        // Draw layer labels
        var inputX = positions.input[0].x;
        var hiddenX = positions.hidden[0].x;
        var outputX = positions.output[0].x;

        this.drawCanvasLabel(ctx, inputX, 25, 'Input (5x5)', 'rgba(99, 102, 241, 0.9)');
        this.drawCanvasLabel(ctx, hiddenX, 25, 'Hidden (' + this.vizConfig.hiddenNodes + ')', 'rgba(139, 92, 246, 0.9)');
        this.drawCanvasLabel(ctx, outputX, 25, 'Output (0-9)', 'rgba(245, 158, 11, 0.9)');
    },

    // ============================================
    // UTILITY FUNCTIONS
    // ============================================

    createSVGElement: function(tag, attrs) {
        var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (var key in attrs) {
            el.setAttribute(key, attrs[key]);
        }
        return el;
    },

    hexToRgb: function(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : { r: 128, g: 128, b: 128 };
    },

    // ============================================
    // METRICS FUNCTIONS
    // ============================================

    loadArchitecture: function() {
        var self = this;
        fetch('/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ getArchitecture: true })
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
            self.displayArchitecture(data.architecture);
        })
        .catch(function(error) {
            console.error('[OCR] Error loading architecture:', error);
        });
    },

    displayArchitecture: function(arch) {
        // Update layer sizes
        document.getElementById('input-size').textContent = arch.architecture.input_layer.size;
        document.getElementById('hidden-size').textContent = arch.architecture.hidden_layer.size;
        document.getElementById('output-size').textContent = arch.architecture.output_layer.size;

        // Update parameters table
        var tbody = document.getElementById('params-tbody');
        var params = arch.parameters;

        tbody.innerHTML = '<tr>' +
            '<td>Theta1 (Input&rarr;Hidden)</td>' +
            '<td>' + params.theta1.shape.join(' x ') + '</td>' +
            '<td>' + params.theta1.count.toLocaleString() + '</td>' +
            '</tr>' +
            '<tr>' +
            '<td>Bias1 (Hidden)</td>' +
            '<td>' + params.b1.shape.join(' x ') + '</td>' +
            '<td>' + params.b1.count.toLocaleString() + '</td>' +
            '</tr>' +
            '<tr>' +
            '<td>Theta2 (Hidden&rarr;Output)</td>' +
            '<td>' + params.theta2.shape.join(' x ') + '</td>' +
            '<td>' + params.theta2.count.toLocaleString() + '</td>' +
            '</tr>' +
            '<tr>' +
            '<td>Bias2 (Output)</td>' +
            '<td>' + params.b2.shape.join(' x ') + '</td>' +
            '<td>' + params.b2.count.toLocaleString() + '</td>' +
            '</tr>' +
            '<tr class="total-row">' +
            '<td colspan="2"><strong>Total Parameters</strong></td>' +
            '<td><strong>' + params.total.toLocaleString() + '</strong></td>' +
            '</tr>';
    },

    evaluateMnist: function() {
        var self = this;
        var sampleSize = parseInt(document.getElementById('mnist-sample-size').value);
        var btn = document.getElementById('eval-mnist-btn');
        var hint = document.getElementById('mnist-hint');

        btn.disabled = true;
        btn.textContent = 'Evaluating...';
        this.showLoading('Running MNIST evaluation...');

        fetch('/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                evaluateMnist: true,
                sampleSize: sampleSize
            })
        })
        .then(function(response) { return response.json(); })
        .then(function(data) {
            self.hideLoading();
            btn.disabled = false;
            btn.textContent = 'Run MNIST Evaluation';
            hint.classList.add('hidden');
            self.displayMnistResults(data.result);
            self.showToast('MNIST evaluation complete: ' + data.result.accuracy_percent + '% accuracy', 'success');
        })
        .catch(function(error) {
            self.hideLoading();
            btn.disabled = false;
            btn.textContent = 'Run MNIST Evaluation';
            console.error('[OCR] MNIST evaluation error:', error);
            self.showToast('MNIST evaluation failed', 'error');
        });
    },

    displayMnistResults: function(result) {
        var resultsDiv = document.getElementById('mnist-results');
        resultsDiv.classList.remove('hidden');

        // Update accuracy display
        document.getElementById('mnist-accuracy').textContent = result.accuracy_percent + '%';
        document.getElementById('mnist-accuracy-label').textContent =
            'Accuracy (' + result.correct + ' / ' + result.sample_size + ' correct)';

        // Create per-digit bars
        var barsContainer = document.getElementById('digit-bars');
        barsContainer.innerHTML = '';

        for (var digit = 0; digit < 10; digit++) {
            var acc = result.per_digit_accuracy[digit.toString()];
            var accPercent = acc !== null ? (acc * 100) : 0;
            var color = acc >= 0.8 ? '#4ade80' : (acc >= 0.5 ? '#fbbf24' : '#f87171');

            var item = document.createElement('div');
            item.className = 'digit-bar-item';
            item.innerHTML =
                '<div class="digit-bar" style="height: ' + accPercent + '%; background-color: ' + color + ';"></div>' +
                '<span class="digit-bar-label">' + digit + '</span>' +
                '<span class="digit-bar-acc">' + (acc !== null ? accPercent.toFixed(0) + '%' : '-') + '</span>';
            barsContainer.appendChild(item);
        }
    }
};

// Initialize on page load
window.addEventListener('DOMContentLoaded', function() {
    ocrDemo.onLoadFunction();
    ocrDemo.loadArchitecture();
});
