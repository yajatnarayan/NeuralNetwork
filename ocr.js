var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH
    BLUE: '#0000FF',
    HOST: 'http://localhost',
    PORT: '8000',
    BATCH_SIZE: 1,
    data: [],
    trainArray: [],
    trainingRequestCount: 0,

    onLoadFunction: function() {
        console.log("onLoadFunction called");
        this.resetCanvas();
        this.drawGrid();

        // Set up event listeners
        var canvas = document.getElementById('canvas');
        if (!canvas) {
            console.error("Canvas element not found!");
            return;
        }

        var ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error("Could not get 2D context for canvas!");
            return;
        }

        console.log("Canvas initialized successfully");

        canvas.addEventListener('mousedown', function(e) {
            ocrDemo.onMouseDown(e, ctx, canvas);
        });

        canvas.addEventListener('mousemove', function(e) {
            ocrDemo.onMouseMove(e, ctx, canvas);
        });

        canvas.addEventListener('mouseup', function(e) {
            ocrDemo.onMouseUp(e, ctx, canvas);
        });

        // Initialize the neural network visualization
        this.initializeVisualization();
        this.loadAndDisplayWeights();
    },

    resetCanvas: function() {
        console.log("resetCanvas called");
        var canvas = document.getElementById('canvas');
        if (!canvas) {
            console.error("Canvas element not found in resetCanvas!");
            return;
        }
        
        var ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error("Could not get 2D context for canvas in resetCanvas!");
            return;
        }
        
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        this.data = [];
        for (var i = 0; i < this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH; i++) {
            this.data.push(0);
        }
        this.drawGrid();
    },

    drawGrid: function() {
        console.log("drawGrid called");
        var canvas = document.getElementById('canvas');
        if (!canvas) {
            console.error("Canvas element not found in drawGrid!");
            return;
        }
        
        var ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error("Could not get 2D context for canvas in drawGrid!");
            return;
        }
        
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; 
                 x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, 
                 y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(e, ctx, canvas) {
        // Stop drawing on the given canvas
        if (canvas) {
            canvas.isDrawing = false;
        }
    },

    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);

        // Guard against out-of-bounds coordinates
        if (xPixel < 0 || yPixel < 0 || xPixel >= this.TRANSLATED_WIDTH || yPixel >= this.TRANSLATED_WIDTH) {
            return;
        }

        // Row-major mapping: row = yPixel, col = xPixel
        var index = yPixel * this.TRANSLATED_WIDTH + xPixel;
        this.data[index] = 1;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH,
            this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },

    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        // Time to send a training batch to the server.
        if (this.trainingRequestCount == this.BATCH_SIZE) {
            alert("Sending training data to server...");
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }

        // Auto-reset canvas and clear input for next training sample
        this.resetCanvas();
        document.getElementById("digit").value = '';
    },

    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit in order to test the network");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);

        // Auto-reset canvas for next test
        this.resetCanvas();
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);

        if (responseJSON.type == "test") {
            console.log("[DATA] Test response received:", responseJSON);

            // Store activations
            if (responseJSON.activations) {
                this.vizState.activations = responseJSON.activations;
                this.vizState.lastPrediction = responseJSON.result;

                console.log("[DATA] Input vector length:", responseJSON.activations.input.length);
                console.log("[DATA] Hidden activations:", responseJSON.activations.hidden.length);
                console.log("[DATA] Output activations:", responseJSON.activations.output);

                // Update all visualizations
                this.renderInputHeatmap(responseJSON.activations.input);
                this.renderConfidenceChart(responseJSON.activations.output, responseJSON.result);
                this.renderNetworkArchitecture();
            }

            alert("The neural network predicts you wrote a '" + responseJSON.result + "'");

        } else if (responseJSON.type == "train" && responseJSON.weights) {
            console.log("[DATA] Train response received");
            // Update stored weights
            this.vizState.weights = responseJSON.weights;
            this.renderNetworkArchitecture();
        }
    },

    onError: function(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        var url = this.HOST + ":" + this.PORT + "/";
        xmlHttp.open('POST', url, true);
        xmlHttp.onload = function() { ocrDemo.receiveResponse(xmlHttp); };
        xmlHttp.onerror = function(e) { ocrDemo.onError(e); };
        var msg = JSON.stringify(json);
        // Use standard content-type header for JSON. Browsers won't let us set Content-length.
        xmlHttp.setRequestHeader('Content-Type', 'application/json');
        xmlHttp.send(msg);
    },

    // ============================================================
    // VISUALIZATION SYSTEM
    // ============================================================

    initializeVisualization: function() {
        console.log("[VIZ] Initializing visualization system");

        this.vizConfig = {
            width: 1200,
            height: 400,
            inputNodes: 400,  // 20x20 pixels
            hiddenNodes: 15,  // Will be updated from server
            outputNodes: 10,
            inputSample: 16,  // Show 4x4 grid of input samples
            layerSpacing: 400,
            nodeRadius: 5
        };

        this.vizState = {
            weights: null,
            activations: null,
            lastPrediction: null
        };

        // Initialize input heatmap canvas
        this.initInputHeatmap();
    },

    initInputHeatmap: function() {
        var canvas = document.getElementById('input-heatmap');
        if (!canvas) {
            console.error("[VIZ] Input heatmap canvas not found");
            return;
        }
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, 200, 200);
        console.log("[VIZ] Input heatmap initialized");
    },

    loadAndDisplayWeights: function() {
        console.log("[VIZ] Loading weights from server");
        var xmlHttp = new XMLHttpRequest();
        var url = this.HOST + ":" + this.PORT + "/";
        xmlHttp.open('POST', url, true);
        xmlHttp.onload = function() {
            console.log("[VIZ] Server response status:", xmlHttp.status);
            if (xmlHttp.status == 200) {
                try {
                    var response = JSON.parse(xmlHttp.responseText);
                    console.log("[VIZ] Response type:", response.type);
                    if (response.type == "weights") {
                        ocrDemo.vizConfig.hiddenNodes = response.hiddenNodes;
                        ocrDemo.vizState.weights = {
                            theta1: response.theta1,
                            theta2: response.theta2
                        };
                        console.log("[VIZ] Weights loaded. Hidden nodes:", response.hiddenNodes);
                        console.log("[VIZ] Theta1 shape:", response.theta1.length, "x", response.theta1[0].length);
                        console.log("[VIZ] Theta2 shape:", response.theta2.length, "x", response.theta2[0].length);
                        ocrDemo.renderNetworkArchitecture();
                    }
                } catch (e) {
                    console.error("[VIZ] Error parsing response:", e);
                }
            } else {
                console.error("[VIZ] HTTP error:", xmlHttp.status, xmlHttp.responseText);
            }
        };
        xmlHttp.onerror = function() {
            console.error("[VIZ] Network error loading weights");
        };
        var msg = JSON.stringify({ getWeights: true });
        xmlHttp.setRequestHeader('Content-Type', 'application/json');
        xmlHttp.send(msg);
    },

    // ============================================================
    // VISUALIZATION RENDERING FUNCTIONS
    // ============================================================

    renderInputHeatmap: function(inputVector) {
        console.log("[VIZ] Rendering input heatmap");
        var canvas = document.getElementById('input-heatmap');
        if (!canvas) {
            console.error("[VIZ] Heatmap canvas not found");
            return;
        }

        var ctx = canvas.getContext('2d');
        var cellSize = 10; // 200px / 20 cells = 10px per cell

        for (var i = 0; i < 20; i++) {
            for (var j = 0; j < 20; j++) {
                var idx = i * 20 + j;
                var value = inputVector[idx];
                var brightness = Math.floor(value * 255);
                ctx.fillStyle = 'rgb(' + brightness + ',' + brightness + ',' + brightness + ')';
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }
        console.log("[VIZ] Heatmap rendered");
    },

    renderConfidenceChart: function(outputActivations, prediction) {
        console.log("[VIZ] Rendering confidence chart");
        var svg = document.getElementById('confidence-chart');
        if (!svg) {
            console.error("[VIZ] Confidence chart not found");
            return;
        }

        svg.innerHTML = '';

        var width = 350;
        var height = 220;
        var barHeight = 18;
        var barSpacing = 2;
        var maxBarWidth = 250;
        var startX = 50;
        var startY = 10;

        // Draw bars for each digit
        for (var i = 0; i < 10; i++) {
            var confidence = outputActivations[i];
            var barWidth = confidence * maxBarWidth;
            var y = startY + i * (barHeight + barSpacing);

            // Background bar
            var bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            bgRect.setAttribute('x', startX);
            bgRect.setAttribute('y', y);
            bgRect.setAttribute('width', maxBarWidth);
            bgRect.setAttribute('height', barHeight);
            bgRect.setAttribute('fill', '#eee');
            bgRect.setAttribute('stroke', '#ccc');
            svg.appendChild(bgRect);

            // Confidence bar
            var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', startX);
            rect.setAttribute('y', y);
            rect.setAttribute('width', barWidth);
            rect.setAttribute('height', barHeight);
            rect.setAttribute('fill', i === prediction ? '#4CAF50' : '#2196F3');
            svg.appendChild(rect);

            // Label
            var label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', startX - 10);
            label.setAttribute('y', y + barHeight/2 + 5);
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('font-size', '14');
            label.setAttribute('font-weight', i === prediction ? 'bold' : 'normal');
            label.textContent = i;
            svg.appendChild(label);

            // Confidence value
            var valueText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            valueText.setAttribute('x', startX + maxBarWidth + 10);
            valueText.setAttribute('y', y + barHeight/2 + 5);
            valueText.setAttribute('font-size', '12');
            valueText.setAttribute('fill', '#666');
            valueText.textContent = (confidence * 100).toFixed(1) + '%';
            svg.appendChild(valueText);
        }

        console.log("[VIZ] Confidence chart rendered. Prediction:", prediction);
    },

    renderNetworkArchitecture: function() {
        console.log("[VIZ] Rendering network architecture");
        var svg = document.getElementById('network-viz');
        if (!svg) {
            console.error("[VIZ] Network SVG not found");
            return;
        }

        if (!this.vizState.weights) {
            console.warn("[VIZ] No weights available yet");
            return;
        }

        svg.innerHTML = '';

        var config = this.vizConfig;
        var weights = this.vizState.weights;
        var theta1 = weights.theta1;  // [hiddenNodes x 400]
        var theta2 = weights.theta2;  // [10 x hiddenNodes]

        var svgWidth = svg.clientWidth || 1200;
        var svgHeight = config.height;

        // Calculate positions for each layer
        var inputX = 100;
        var hiddenX = svgWidth / 2;
        var outputX = svgWidth - 100;

        // Sample input nodes (4x4 grid from 20x20)
        var inputSampleIndices = [];
        for (var i = 0; i < 4; i++) {
            for (var j = 0; j < 4; j++) {
                inputSampleIndices.push((i * 5) * 20 + (j * 5));
            }
        }

        var inputPositions = this.calculateNodePositions(config.inputSample, svgHeight, inputX);
        var hiddenPositions = this.calculateNodePositions(config.hiddenNodes, svgHeight, hiddenX);
        var outputPositions = this.calculateNodePositions(config.outputNodes, svgHeight, outputX);

        // Draw connections (only strong ones for clarity)
        this.drawConnections(svg, inputPositions, hiddenPositions, theta1, inputSampleIndices, true);
        this.drawConnections(svg, hiddenPositions, outputPositions, theta2, null, false);

        // Draw nodes with activations if available
        var inputActivations = this.vizState.activations ? this.vizState.activations.input : null;
        var hiddenActivations = this.vizState.activations ? this.vizState.activations.hidden : null;
        var outputActivations = this.vizState.activations ? this.vizState.activations.output : null;

        this.drawNodes(svg, inputPositions, '#4A90E2', inputActivations, inputSampleIndices);
        this.drawNodes(svg, hiddenPositions, '#50C878', hiddenActivations, null);
        this.drawNodes(svg, outputPositions, '#E94B3C', outputActivations, null);

        // Add layer labels
        this.addLabel(svg, inputX, 20, 'Input (4×4 sample)');
        this.addLabel(svg, hiddenX, 20, 'Hidden (' + config.hiddenNodes + ')');
        this.addLabel(svg, outputX, 20, 'Output (0-9)');

        console.log("[VIZ] Network architecture rendered");
    },

    // ============================================================
    // HELPER FUNCTIONS FOR NETWORK VISUALIZATION
    // ============================================================

    calculateNodePositions: function(numNodes, height, x) {
        var positions = [];
        var spacing = (height - 80) / (numNodes + 1);
        for (var i = 0; i < numNodes; i++) {
            positions.push({ x: x, y: 40 + spacing * (i + 1) });
        }
        return positions;
    },

    drawConnections: function(svg, fromPositions, toPositions, weights, sampleIndices, isInputLayer) {
        // Find max weight for normalization
        var maxWeight = 0.001; // Avoid division by zero
        for (var i = 0; i < weights.length; i++) {
            for (var j = 0; j < weights[i].length; j++) {
                maxWeight = Math.max(maxWeight, Math.abs(weights[i][j]));
            }
        }

        console.log("[VIZ] Drawing connections. Max weight:", maxWeight);

        // Only draw strong connections to avoid clutter
        var threshold = 0.15;
        var drawnCount = 0;

        for (var i = 0; i < toPositions.length; i++) {
            for (var j = 0; j < fromPositions.length; j++) {
                var actualFromIdx = isInputLayer && sampleIndices ? sampleIndices[j] : j;
                var weight = weights[i][actualFromIdx];
                var normalizedWeight = weight / maxWeight;

                // Only draw significant connections
                if (Math.abs(normalizedWeight) > threshold) {
                    var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', fromPositions[j].x);
                    line.setAttribute('y1', fromPositions[j].y);
                    line.setAttribute('x2', toPositions[i].x);
                    line.setAttribute('y2', toPositions[i].y);

                    // Color: green for positive, red for negative
                    var color = weight > 0 ? '#00aa00' : '#cc0000';
                    var opacity = Math.abs(normalizedWeight) * 0.6;
                    var strokeWidth = Math.abs(normalizedWeight) * 2 + 0.5;

                    line.setAttribute('stroke', color);
                    line.setAttribute('stroke-opacity', opacity);
                    line.setAttribute('stroke-width', strokeWidth);

                    svg.appendChild(line);
                    drawnCount++;
                }
            }
        }

        console.log("[VIZ] Drew", drawnCount, "connections");
    },

    drawNodes: function(svg, positions, baseColor, activations, sampleIndices) {
        for (var i = 0; i < positions.length; i++) {
            // Get activation value if available
            var activation = null;
            if (activations) {
                var actualIdx = sampleIndices ? sampleIndices[i] : i;
                activation = activations[actualIdx];
            }

            var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', positions[i].x);
            circle.setAttribute('cy', positions[i].y);
            circle.setAttribute('r', this.vizConfig.nodeRadius);

            // If we have activation data, modulate the color intensity
            if (activation !== null && activation !== undefined) {
                // Higher activation = brighter/more saturated
                var intensity = Math.min(1, Math.max(0, activation));
                var colorValue = Math.floor(255 * intensity);

                // Extract RGB from base color and blend
                var rgb = this.hexToRgb(baseColor);
                var finalColor = 'rgb(' +
                    Math.floor(rgb.r * intensity + 240 * (1 - intensity)) + ',' +
                    Math.floor(rgb.g * intensity + 240 * (1 - intensity)) + ',' +
                    Math.floor(rgb.b * intensity + 240 * (1 - intensity)) + ')';

                circle.setAttribute('fill', finalColor);
                circle.setAttribute('stroke', baseColor);
                circle.setAttribute('stroke-width', 2);
            } else {
                circle.setAttribute('fill', baseColor);
                circle.setAttribute('stroke', '#333');
                circle.setAttribute('stroke-width', 1);
            }

            svg.appendChild(circle);
        }
    },

    hexToRgb: function(hex) {
        var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : {r: 128, g: 128, b: 128};
    },

    addLabel: function(svg, x, y, text) {
        var label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x);
        label.setAttribute('y', y);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '13');
        label.setAttribute('font-weight', 'bold');
        label.setAttribute('fill', '#333');
        label.textContent = text;
        svg.appendChild(label);
    }
};

// Set up event listeners when the page loads
window.onload = function() {
    console.log("Window onload event fired");
    ocrDemo.onLoadFunction();
};