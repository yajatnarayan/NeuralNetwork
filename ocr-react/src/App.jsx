import { useState, useEffect } from 'react'
import DrawingCanvas from './components/DrawingCanvas'
import InputHeatmap from './components/InputHeatmap'
import ConfidenceChart from './components/ConfidenceChart'
import NetworkVisualization from './components/NetworkVisualization'
import MetricsPanel from './components/MetricsPanel'
import { trainNetwork, testNetwork, getWeights } from './api/client'
import './App.css'

console.log('[App] Module loaded - OCR Neural Network Demo initializing')

function App() {
  console.log('[App] App component rendering')

  const [inputVector, setInputVector] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [activations, setActivations] = useState(null)
  const [weights, setWeights] = useState(null)
  const [digit, setDigit] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  console.log('[App] Current state:', {
    hasInputVector: !!inputVector,
    prediction,
    hasActivations: !!activations,
    hasWeights: !!weights,
    digit,
    isLoading,
    error
  })

  // Load weights on mount
  useEffect(() => {
    console.log('[App] useEffect - component mounted, loading weights')
    loadWeights()
  }, [])

  const loadWeights = async () => {
    console.log('[App] loadWeights() called')
    try {
      const data = await getWeights()
      console.log('[App] Weights loaded successfully:', {
        hiddenNodes: data.hiddenNodes,
        theta1Shape: data.theta1 ? `[${data.theta1.length}x${data.theta1[0]?.length}]` : 'missing',
        theta2Shape: data.theta2 ? `[${data.theta2.length}x${data.theta2[0]?.length}]` : 'missing'
      })
      setWeights(data)
    } catch (err) {
      console.error('[App] loadWeights FAILED:', err.message)
      console.error('[App] Full error:', err)
      setError('Failed to load network weights')
    }
  }

  const handleTrain = async (drawnData) => {
    console.log('[App] handleTrain() called')
    console.log('[App] Training input:', {
      digit,
      drawnDataLength: drawnData?.length,
      nonZeroPixels: drawnData?.filter(v => v > 0).length,
      allZeros: drawnData?.every(v => v === 0)
    })

    if (!digit || drawnData.every(v => v === 0)) {
      console.warn('[App] Training aborted: missing digit or empty drawing')
      alert('Please draw a digit and enter its value (0-9)')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      console.log('[App] Calling trainNetwork API...')
      const response = await trainNetwork([{
        y0: drawnData,
        label: parseInt(digit)
      }])

      console.log('[App] Training response:', {
        hasWeights: !!response.weights,
        responseKeys: Object.keys(response)
      })

      if (response.weights) {
        setWeights(response.weights)
        console.log('[App] Training complete, weights updated')
      }

      setDigit('') // Clear input after training
    } catch (err) {
      console.error('[App] Training FAILED:', err.message)
      console.error('[App] Full error:', err)
      setError('Training failed')
    } finally {
      setIsLoading(false)
      console.log('[App] Training process finished')
    }
  }

  const handleTest = async (drawnData) => {
    console.log('[App] handleTest() called')
    console.log('[App] Test input:', {
      drawnDataLength: drawnData?.length,
      nonZeroPixels: drawnData?.filter(v => v > 0).length,
      allZeros: drawnData?.every(v => v === 0)
    })

    if (drawnData.every(v => v === 0)) {
      console.warn('[App] Test aborted: empty drawing')
      alert('Please draw a digit to test')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      console.log('[App] Calling testNetwork API...')
      const response = await testNetwork(drawnData)

      console.log('[App] Test response:', {
        result: response.result,
        hasActivations: !!response.activations,
        activationKeys: response.activations ? Object.keys(response.activations) : [],
        inputLength: response.activations?.input?.length,
        hiddenLength: response.activations?.hidden?.length,
        outputLength: response.activations?.output?.length
      })

      setInputVector(response.activations.input)
      setActivations(response.activations)
      setPrediction(response.result)

      console.log('[App] Prediction:', response.result)
      alert(`Neural network predicts: ${response.result}`)
    } catch (err) {
      console.error('[App] Test FAILED:', err.message)
      console.error('[App] Full error:', err)
      setError('Prediction failed')
    } finally {
      setIsLoading(false)
      console.log('[App] Test process finished')
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
