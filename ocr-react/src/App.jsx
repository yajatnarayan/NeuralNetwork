import { useState, useEffect } from 'react'
import DrawingCanvas from './components/DrawingCanvas'
import InputHeatmap from './components/InputHeatmap'
import ConfidenceChart from './components/ConfidenceChart'
import NetworkVisualization from './components/NetworkVisualization'
import MetricsPanel from './components/MetricsPanel'
import { trainNetwork, testNetwork, getWeights } from './api/client'
import './App.css'

function App() {
  const [inputVector, setInputVector] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [activations, setActivations] = useState(null)
  const [weights, setWeights] = useState(null)
  const [digit, setDigit] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load weights on mount
  useEffect(() => {
    loadWeights()
  }, [])

  const loadWeights = async () => {
    try {
      const data = await getWeights()
      setWeights(data)
      console.log('[APP] Weights loaded:', data.hiddenNodes, 'hidden nodes')
    } catch (err) {
      setError('Failed to load network weights')
      console.error('[APP] Error loading weights:', err)
    }
  }

  const handleTrain = async (drawnData) => {
    if (!digit || drawnData.every(v => v === 0)) {
      alert('Please draw a digit and enter its value (0-9)')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await trainNetwork([{
        y0: drawnData,
        label: parseInt(digit)
      }])

      if (response.weights) {
        setWeights(response.weights)
        console.log('[APP] Training complete, weights updated')
      }

      setDigit('') // Clear input after training
    } catch (err) {
      setError('Training failed')
      console.error('[APP] Training error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleTest = async (drawnData) => {
    if (drawnData.every(v => v === 0)) {
      alert('Please draw a digit to test')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const response = await testNetwork(drawnData)

      setInputVector(response.activations.input)
      setActivations(response.activations)
      setPrediction(response.result)

      console.log('[APP] Prediction:', response.result)
      alert(`Neural network predicts: ${response.result}`)
    } catch (err) {
      setError('Prediction failed')
      console.error('[APP] Prediction error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>OCR Neural Network Demo</h1>
        {error && <div className="error-banner">{error}</div>}
      </header>

      <div className="content">
        {/* Top Row: Input Controls */}
        <div className="top-row">
          <div className="card">
            <h3>Drawing Canvas (20×20)</h3>
            <DrawingCanvas
              onTrain={handleTrain}
              onTest={handleTest}
              disabled={isLoading}
            />
            <div className="controls">
              <input
                type="text"
                placeholder="Digit (0-9)"
                value={digit}
                onChange={(e) => setDigit(e.target.value)}
                maxLength={1}
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="card">
            <h3>Input Vector (400 dims)</h3>
            <InputHeatmap data={inputVector} />
            <p className="caption">Normalized 20×20 grid</p>
          </div>

          <div className="card">
            <h3>Prediction Confidence</h3>
            <ConfidenceChart
              activations={activations?.output}
              prediction={prediction}
            />
          </div>
        </div>

        {/* Bottom Row: Network Visualization */}
        <div className="bottom-row">
          <div className="card full-width">
            <h3>Neural Network Architecture (Input → Hidden → Output)</h3>
            <NetworkVisualization
              weights={weights}
              activations={activations}
            />
            <div className="legend">
              <span className="legend-item">
                <span className="dot positive"></span> Positive weights
              </span>
              <span className="legend-item">
                <span className="dot negative"></span> Negative weights
              </span>
              <span className="legend-item">
                <span className="dot input"></span> Input layer
              </span>
              <span className="legend-item">
                <span className="dot hidden"></span> Hidden layer
              </span>
              <span className="legend-item">
                <span className="dot output"></span> Output layer
              </span>
            </div>
          </div>
        </div>

        {/* Metrics Section */}
        <div className="metrics-row">
          <div className="card full-width">
            <h3>Model Metrics</h3>
            <MetricsPanel />
          </div>
        </div>
      </div>

      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
        </div>
      )}
    </div>
  )
}

export default App
