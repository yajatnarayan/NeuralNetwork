import './ConfidenceChart.css'

const WIDTH = 350
const HEIGHT = 220
const BAR_HEIGHT = 18
const BAR_SPACING = 2
const MAX_BAR_WIDTH = 250
const START_X = 50
const START_Y = 10

console.log('[ConfidenceChart] Module loaded with config:', { WIDTH, HEIGHT, MAX_BAR_WIDTH })

function ConfidenceChart({ activations, prediction }) {
  console.log('[ConfidenceChart] Rendering with:', {
    hasActivations: !!activations,
    activationsLength: activations?.length,
    activationsType: typeof activations,
    isArray: Array.isArray(activations),
    prediction,
    activationValues: activations?.map((v, i) => ({ digit: i, confidence: v?.toFixed(4) }))
  })

  if (!activations || activations.length !== 10) {
    console.log('[ConfidenceChart] No valid activations, showing placeholder')
    return (
      <svg width={WIDTH} height={HEIGHT} className="confidence-chart">
        <text x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" fill="#999">
          Draw and test a digit to see confidence
        </text>
      </svg>
    )
  }

  console.log('[ConfidenceChart] Drawing bars for', activations.length, 'digits')

  return (
    <svg width={WIDTH} height={HEIGHT} className="confidence-chart">
      {activations.map((confidence, i) => {
        const barWidth = confidence * MAX_BAR_WIDTH
        const y = START_Y + i * (BAR_HEIGHT + BAR_SPACING)
        const isPrediction = i === prediction

        return (
          <g key={i}>
            {/* Background bar */}
            <rect
              x={START_X}
              y={y}
              width={MAX_BAR_WIDTH}
              height={BAR_HEIGHT}
              fill="#eee"
              stroke="#ccc"
            />

            {/* Confidence bar */}
            <rect
              x={START_X}
              y={y}
              width={barWidth}
              height={BAR_HEIGHT}
              fill={isPrediction ? '#4CAF50' : '#2196F3'}
            />

            {/* Digit label */}
            <text
              x={START_X - 10}
              y={y + BAR_HEIGHT / 2 + 5}
              textAnchor="end"
              fontSize="14"
              fontWeight={isPrediction ? 'bold' : 'normal'}
            >
              {i}
            </text>

            {/* Confidence percentage */}
            <text
              x={START_X + MAX_BAR_WIDTH + 10}
              y={y + BAR_HEIGHT / 2 + 5}
              fontSize="12"
              fill="#666"
            >
              {(confidence * 100).toFixed(1)}%
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export default ConfidenceChart
