import json
import http.server
import socketserver
import os
from ocr import OCRNeuralNetwork

# Initialize the OCR neural network
nn = OCRNeuralNetwork(15, None, None, None, True)


class OCRRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for OCR neural network operations."""

    def send_cors_headers(self):
        """Send CORS headers to allow cross-origin requests."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests for static files."""
        print(f"GET request for: {self.path}")

        # Serve the HTML file for the root path
        if self.path == '/':
            self.path = '/ocr.html'

        # Serve static files (JS, CSS, HTML)
        if self.path.endswith(('.js', '.css', '.html')):
            try:
                file_path = self.path.lstrip('/')
                if not os.path.exists(file_path):
                    self.send_error(404, f"File not found: {self.path}")
                    return

                with open(file_path, 'rb') as file:
                    content = file.read()

                self.send_response(200)

                # Set appropriate content type
                content_types = {
                    '.js': 'application/javascript',
                    '.css': 'text/css',
                    '.html': 'text/html'
                }
                ext = os.path.splitext(self.path)[1]
                self.send_header('Content-type', content_types.get(ext, 'text/plain'))
                self.send_header('Content-Length', len(content))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                print(f"Error serving file {self.path}: {e}")
                self.send_error(500, f"Error serving file: {str(e)}")
                return

        # Default behavior for other files
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        """Handle POST requests for training and prediction."""
        print("POST request received")
        response_code = 200
        response = {}

        try:
            # Safely get content length
            content_length = self.headers.get('Content-Length')
            if content_length is None:
                self.send_json_response(400, {"error": "Missing Content-Length header"})
                return

            var_len = int(content_length)
            if var_len <= 0:
                self.send_json_response(400, {"error": "Empty request body"})
                return

            content = self.rfile.read(var_len)

            try:
                payload = json.loads(content)
            except json.JSONDecodeError as e:
                self.send_json_response(400, {"error": f"Invalid JSON: {str(e)}"})
                return

            # Handle training request
            if payload.get('train'):
                train_array = payload.get('trainArray', [])
                if not train_array:
                    self.send_json_response(400, {"error": "No training data provided"})
                    return

                # Validate training data
                for item in train_array:
                    if 'y0' not in item or 'label' not in item:
                        self.send_json_response(400, {"error": "Invalid training data format"})
                        return
                    if len(item['y0']) != 400:
                        self.send_json_response(400, {"error": "Input vector must be 400 dimensions"})
                        return
                    if not (0 <= item['label'] <= 9):
                        self.send_json_response(400, {"error": "Label must be between 0 and 9"})
                        return

                print(f"Training with {len(train_array)} samples")
                nn.train(train_array)
                nn.save()

                response = {
                    "type": "train",
                    "weights": {
                        "theta1": nn.theta1.tolist(),
                        "theta2": nn.theta2.tolist()
                    }
                }

            # Handle prediction request
            elif payload.get('predict'):
                image = payload.get('image')
                if image is None:
                    self.send_json_response(400, {"error": "No image data provided"})
                    return

                if len(image) != 400:
                    self.send_json_response(400, {"error": "Image must be 400 pixels (20x20)"})
                    return

                print("Processing prediction request")
                activations_data = nn.predict(image, return_activations=True)

                response = {
                    "type": "test",
                    "result": activations_data['prediction'],
                    "activations": {
                        "input": activations_data['input_vector'],
                        "hidden": activations_data['hidden_activations'],
                        "output": activations_data['output_activations']
                    }
                }

            # Handle get weights request
            elif payload.get('getWeights'):
                print("Returning network weights")
                response = {
                    "type": "weights",
                    "theta1": nn.theta1.tolist(),
                    "theta2": nn.theta2.tolist(),
                    "hiddenNodes": nn.num_hidden_nodes
                }

            else:
                self.send_json_response(400, {"error": "Unknown request type"})
                return

            self.send_json_response(response_code, response)

        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            self.send_json_response(500, {"error": f"Server error: {str(e)}"})

    def send_json_response(self, code, data):
        """Send a JSON response with proper headers."""
        response_body = json.dumps(data).encode('utf-8')

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response_body))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        """Override to provide cleaner logging."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    PORT = 8000
    print(f"Starting OCR Neural Network Server")
    print(f"=" * 50)
    print(f"Port: {PORT}")
    print(f"Directory: {os.getcwd()}")
    print(f"Neural network: {nn.num_hidden_nodes} hidden nodes")
    print(f"=" * 50)
    print(f"\nOpen http://localhost:{PORT} in your browser")
    print(f"Press Ctrl+C to stop the server\n")

    # Allow immediate reuse of the address
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), OCRRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
            print("Server stopped.")


if __name__ == "__main__":
    main()
