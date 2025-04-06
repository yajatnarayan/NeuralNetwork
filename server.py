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
        elif payload.get('predict'):
            try:
                print("Prediction request received")
                response = {
                    "type": "test", 
                    "result": nn.predict(str(payload['image']))
                }
            except Exception as e:
                print(f"Error during prediction: {e}")
                response_code = 500
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
    PORT = 5000
    print(f"Starting server on port {PORT}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    with socketserver.TCPServer(("", PORT), OCRRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        httpd.serve_forever()