import { useState, useEffect } from 'react'
import { getArchitecture, evaluateMnist } from '../api/client'

function MetricsPanel() {
  const [architecture, setArchitecture] = useState(null)
  const [mnistResult, setMnistResult] = useState(null)
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [sampleSize, setSampleSize] = useState(1000)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadArchitecture()
  }, [])

  const loadArchitecture = async () => {
    try {
      const data = await getArchitecture()
      setArchitecture(data.architecture)
    } catch (err) {
      setError('Failed to load architecture')
      console.error('[MetricsPanel] Error loading architecture:', err)
    }
  }

  const handleEvaluateMnist = async () => {
    setIsEvaluating(true)
    setError(null)

    try {
      const data = await evaluateMnist(sampleSize)
      setMnistResult(data.result)
      setArchitecture(data.architecture)
    } catch (err) {
      setError('MNIST evaluation failed')
      console.error('[MetricsPanel] Error evaluating MNIST:', err)
    } finally {
      setIsEvaluating(false)
    }
  }

  const formatNumber = (num) => {
    return num?.toLocaleString() ?? '-'
  }

  return (
    <div className="metrics-panel">
      {/* Network Architecture Section */}
      <div className="metrics-section">
        <h4>Network Architecture</h4>
        {architecture ? (
          <div className="architecture-info">
            <div className="arch-diagram">
              <div className="layer input-layer">
                <span className="layer-size">{architecture.architecture.input_layer.size}</span>
                <span className="layer-label">Input</span>
              </div>
              <div className="layer-arrow">&rarr;</div>
              <div className="layer hidden-layer">
                <span className="layer-size">{architecture.architecture.hidden_layer.size}</span>
                <span className="layer-label">Hidden</span>
              </div>
              <div className="layer-arrow">&rarr;</div>
              <div className="layer output-layer">
                <span className="layer-size">{architecture.architecture.output_layer.size}</span>
                <span className="layer-label">Output</span>
              </div>
            </div>

            <div className="params-table">
              <h5>Parameters</h5>
              <table>
                <thead>
                  <tr>
                    <th>Layer</th>
                    <th>Shape</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Theta1 (Input&rarr;Hidden)</td>
                    <td>{architecture.parameters.theta1.shape.join(' x ')}</td>
                    <td>{formatNumber(architecture.parameters.theta1.count)}</td>
                  </tr>
                  <tr>
                    <td>Bias1 (Hidden)</td>
                    <td>{architecture.parameters.b1.shape.join(' x ')}</td>
                    <td>{formatNumber(architecture.parameters.b1.count)}</td>
                  </tr>
                  <tr>
                    <td>Theta2 (Hidden&rarr;Output)</td>
                    <td>{architecture.parameters.theta2.shape.join(' x ')}</td>
                    <td>{formatNumber(architecture.parameters.theta2.count)}</td>
                  </tr>
                  <tr>
                    <td>Bias2 (Output)</td>
                    <td>{architecture.parameters.b2.shape.join(' x ')}</td>
                    <td>{formatNumber(architecture.parameters.b2.count)}</td>
                  </tr>
                  <tr className="total-row">
                    <td colSpan="2"><strong>Total Parameters</strong></td>
                    <td><strong>{formatNumber(architecture.parameters.total)}</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <p className="loading-text">Loading architecture...</p>
        )}
      </div>

      {/* MNIST Evaluation Section */}
      <div className="metrics-section">
        <h4>MNIST Test Accuracy</h4>
        <div className="mnist-controls">
          <label>
            Sample Size:
            <select
              value={sampleSize}
              onChange={(e) => setSampleSize(parseInt(e.target.value))}
              disabled={isEvaluating}
            >
              <option value={100}>100 (fast)</option>
              <option value={500}>500</option>
              <option value={1000}>1,000</option>
              <option value={2000}>2,000</option>
              <option value={5000}>5,000</option>
              <option value={10000}>10,000 (full test set)</option>
            </select>
          </label>
          <button
            onClick={handleEvaluateMnist}
            disabled={isEvaluating}
            className="evaluate-btn"
          >
            {isEvaluating ? 'Evaluating...' : 'Run MNIST Evaluation'}
          </button>
        </div>

        {error && <p className="error-text">{error}</p>}

        {mnistResult && (
          <div className="mnist-results">
            <div className="accuracy-display">
              <div className="accuracy-value">
                {mnistResult.accuracy_percent}%
              </div>
              <div className="accuracy-label">
                Accuracy ({mnistResult.correct} / {mnistResult.sample_size} correct)
              </div>
            </div>

            <div className="per-digit-accuracy">
              <h5>Per-Digit Accuracy</h5>
              <div className="digit-bars">
                {Object.entries(mnistResult.per_digit_accuracy).map(([digit, acc]) => (
                  <div key={digit} className="digit-bar-container">
                    <div
                      className="digit-bar"
                      style={{
                        height: `${(acc ?? 0) * 100}%`,
                        backgroundColor: acc >= 0.8 ? '#4ade80' : acc >= 0.5 ? '#fbbf24' : '#f87171'
                      }}
                    />
                    <span className="digit-label">{digit}</span>
                    <span className="digit-acc">{acc !== null ? `${(acc * 100).toFixed(0)}%` : '-'}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {!mnistResult && !isEvaluating && (
          <p className="hint-text">
            Click "Run MNIST Evaluation" to test the model on the MNIST test dataset.
          </p>
        )}
      </div>
    </div>
  )
}

export default MetricsPanel
