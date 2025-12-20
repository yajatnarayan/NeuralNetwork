import json
import http.server
import socketserver
import os
from ocr import OCRNeuralNetwork

# Initialize the OCR neural network
nn = OCRNeuralNetwork(15, None, None, None, True)

class OCRRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"GET request for: {self.path}")
        
        # Serve the HTML file for the root path
        if self.path == '/':
            self.path = '/ocr.html'
        
        # Serve static files (JS, CSS)
        if self.path.endswith('.js') or self.path.endswith('.css'):
            try:
                file_path = self.path.lstrip('/')
                with open(file_path, 'rb') as file:
                    self.send_response(200)
                    if self.path.endswith('.js'):
                        self.send_header('Content-type', 'application/javascript')
                    elif self.path.endswith('.css'):
                        self.send_header('Content-type', 'text/css')
                    self.end_headers()
                    self.wfile.write(file.read())
                return
            except Exception as e:
                print(f"Error serving file {self.path}: {e}")
                self.send_error(404, f"File not found: {self.path}")
                return
        
        # Default behavior for other files
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        print(f"POST request received")
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content)

        if payload.get('train'):
            print("Training request received")
            nn.train(payload['trainArray'])
            nn.save()
            # Return updated weights for visualization
            response = {
                "type": "train",
                "weights": {
                    "theta1": nn.theta1.tolist(),
                    "theta2": nn.theta2.tolist()
                }
            }
        elif payload.get('predict'):
            try:
                print("Prediction request received")
                # Expect payload['image'] to be a list/array of 400 values
                activations_data = nn.predict(payload['image'], return_activations=True)
                response = {
                    "type": "test",
                    "result": activations_data['prediction'],
                    "activations": {
                        "input": activations_data['input_vector'],
                        "hidden": activations_data['hidden_activations'],
                        "output": activations_data['output_activations']
                    }
                }
            except Exception as e:
                print(f"Error during prediction: {e}")
                response_code = 500
        elif payload.get('getWeights'):
            print("Get weights request received")
            # Return current network weights
            response = {
                "type": "weights",
                "theta1": nn.theta1.tolist(),
                "theta2": nn.theta2.tolist(),
                "hiddenNodes": nn.num_hidden_nodes
            }
        else:
            print("Invalid request received")
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode())
        return

if __name__ == "__main__":
    PORT = 8000
    print(f"Starting server on port {PORT}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    # Allow immediate reuse of the address to avoid "Address already in use"
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", PORT), OCRRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down server")
            httpd.server_close()