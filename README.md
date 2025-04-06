# Optical Character Recognition (OCR) System

This is a simple OCR system that can recognize handwritten digits using an Artificial Neural Network (ANN).

## Overview

The system consists of:
- A web-based interface for drawing digits
- A client-side JavaScript component for handling user interactions
- A Python server that processes requests
- A neural network implementation for OCR

## Requirements

- Python 3.6+
- NumPy
- A web browser

## Installation

1. Install the required Python packages:
```
pip install numpy
```

## Usage

1. Start the server:
```
python server.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Draw a digit on the canvas and:
   - Enter the digit value in the text field and click "Train" to train the network
   - Click "Test" to have the network predict the digit
   - Click "Reset" to clear the canvas

## How It Works

1. The user draws a digit on the canvas
2. The drawing is converted to a 20x20 pixel representation (400 values)
3. This data is sent to the server
4. The neural network processes the data and returns a prediction

## Neural Network Design

The neural network design can be experimented with using:
```
python neural_network_design.py
```

This will test different numbers of hidden nodes and report the performance of each configuration.

## Architecture

- `ocr.html`: The web interface
- `ocr.js`: Client-side JavaScript for handling user interactions
- `ocr.css`: Styling for the web interface
- `server.py`: Python server that handles requests
- `ocr.py`: Neural network implementation
- `neural_network_design.py`: Script for experimenting with network design
