import { useMemo } from 'react'
import './NetworkVisualization.css'

const SVG_HEIGHT = 400
const NODE_RADIUS = 5
const THRESHOLD = 0.15

console.log('[NetworkVisualization] Module loaded with config:', { SVG_HEIGHT, NODE_RADIUS, THRESHOLD })

function NetworkVisualization({ weights, activations }) {
  console.log('[NetworkVisualization] Rendering with:', {
    hasWeights: !!weights,
    theta1Shape: weights?.theta1 ? `[${weights.theta1.length}x${weights.theta1[0]?.length}]` : 'none',
    theta2Shape: weights?.theta2 ? `[${weights.theta2.length}x${weights.theta2[0]?.length}]` : 'none',
    hiddenNodes: weights?.hiddenNodes,
    hasActivations: !!activations,
    activationKeys: activations ? Object.keys(activations) : []
  })

  const visualization = useMemo(() => {
    console.log('[NetworkVisualization] useMemo recalculating visualization')
    if (!weights) {
      console.log('[NetworkVisualization] No weights, returning null')
      return null
    }

    const { theta1, theta2, hiddenNodes } = weights

    // Validate weights structure
    if (!theta1 || !theta2 || !Array.isArray(theta1) || !Array.isArray(theta2)) {
      console.error('[NetworkVisualization] Invalid weights structure:', { theta1: typeof theta1, theta2: typeof theta2 })
      return null
    }

    // Use actual array lengths for safety
    const actualHiddenNodes = theta1.length
    const actualOutputNodes = theta2.length

    console.log('[NetworkVisualization] Weight dimensions:', {
      theta1Rows: theta1.length,
      theta1Cols: theta1[0]?.length,
      theta2Rows: theta2.length,
      theta2Cols: theta2[0]?.length,
      declaredHiddenNodes: hiddenNodes,
      actualHiddenNodes,
      actualOutputNodes
    })

    const width = 1200
    const height = SVG_HEIGHT

    // Layer positions
    const inputX = 100
    const hiddenX = width / 2
    const outputX = width - 100

    // Sample 4x4 grid from 20x20 input
    const inputSampleIndices = []
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        inputSampleIndices.push((i * 5) * 20 + (j * 5))
      }
    }

    const inputPos = calculatePositions(16, height, inputX)
    const hiddenPos = calculatePositions(actualHiddenNodes, height, hiddenX)
    const outputPos = calculatePositions(actualOutputNodes, height, outputX)

    // Calculate connections
    const connections = []

    // Input to Hidden - with bounds checking
    const maxWeight1 = Math.max(...theta1.flat().map(Math.abs))
    if (maxWeight1 > 0) {
      theta1.forEach((hiddenWeights, hi) => {
        if (!hiddenPos[hi]) {
          console.warn('[NetworkVisualization] Missing hiddenPos for index:', hi)
          return
        }
        inputSampleIndices.forEach((inputIdx, ii) => {
          if (!inputPos[ii] || !hiddenWeights) return

          const weight = hiddenWeights[inputIdx]
          if (weight === undefined) return

          const norm = weight / maxWeight1

          if (Math.abs(norm) > THRESHOLD) {
            connections.push({
              x1: inputPos[ii].x,
              y1: inputPos[ii].y,
              x2: hiddenPos[hi].x,
              y2: hiddenPos[hi].y,
              color: weight > 0 ? '#00aa00' : '#cc0000',
              opacity: Math.abs(norm) * 0.6,
              width: Math.abs(norm) * 2 + 0.5
            })
          }
        })
      })
    }

    // Hidden to Output - with bounds checking
    const maxWeight2 = Math.max(...theta2.flat().map(Math.abs))
    if (maxWeight2 > 0) {
      theta2.forEach((outputWeights, oi) => {
        if (!outputPos[oi] || !outputWeights) return

        outputWeights.forEach((weight, hi) => {
          if (!hiddenPos[hi] || weight === undefined) return

          const norm = weight / maxWeight2

          if (Math.abs(norm) > THRESHOLD) {
            connections.push({
              x1: hiddenPos[hi].x,
              y1: hiddenPos[hi].y,
              x2: outputPos[oi].x,
              y2: outputPos[oi].y,
              color: weight > 0 ? '#00aa00' : '#cc0000',
              opacity: Math.abs(norm) * 0.6,
              width: Math.abs(norm) * 2 + 0.5
            })
          }
        })
      })
    }

    // Node data
    const nodes = [
      ...inputPos.map((pos, i) => ({
        ...pos,
        color: '#4A90E2',
        activation: activations?.input?.[inputSampleIndices[i]]
      })),
      ...hiddenPos.map((pos, i) => ({
        ...pos,
        color: '#50C878',
        activation: activations?.hidden?.[i]
      })),
      ...outputPos.map((pos, i) => ({
        ...pos,
        color: '#E94B3C',
        activation: activations?.output?.[i]
      }))
    ]

    console.log('[NetworkVisualization] Visualization computed:', {
      connectionCount: connections.length,
      nodeCount: nodes.length,
      inputNodes: inputPos.length,
      hiddenNodes: hiddenPos.length,
      outputNodes: outputPos.length
    })

    return { connections, nodes, inputPos, hiddenPos, outputPos, hiddenNodes: actualHiddenNodes }
  }, [weights, activations])

  if (!visualization) {
    return (
      <svg width="100%" height={SVG_HEIGHT} className="network-viz">
        <text x="50%" y="50%" textAnchor="middle" fill="#999">
          Loading network visualization...
        </text>
      </svg>
    )
  }

  const { connections, nodes, inputPos, hiddenPos, outputPos, hiddenNodes } = visualization

  return (
    <svg width="100%" height={SVG_HEIGHT} className="network-viz" viewBox="0 0 1200 400">
      {/* Draw connections first */}
      {connections.map((conn, i) => (
        <line
          key={`conn-${i}`}
          x1={conn.x1}
          y1={conn.y1}
          x2={conn.x2}
          y2={conn.y2}
          stroke={conn.color}
          strokeOpacity={conn.opacity}
          strokeWidth={conn.width}
        />
      ))}

      {/* Draw nodes */}
      {nodes.map((node, i) => {
        let fill = node.color
        let stroke = '#333'
        let strokeWidth = 1

        // Modulate color by activation if available
        if (node.activation !== undefined) {
          const intensity = Math.min(1, Math.max(0, node.activation))
          const rgb = hexToRgb(node.color)
          fill = `rgb(${Math.floor(rgb.r * intensity + 240 * (1 - intensity))},${Math.floor(rgb.g * intensity + 240 * (1 - intensity))},${Math.floor(rgb.b * intensity + 240 * (1 - intensity))})`
          stroke = node.color
          strokeWidth = 2
        }

        return (
          <circle
            key={`node-${i}`}
            cx={node.x}
            cy={node.y}
            r={NODE_RADIUS}
            fill={fill}
            stroke={stroke}
            strokeWidth={strokeWidth}
          />
        )
      })}

      {/* Layer labels */}
      <text x={inputPos[0]?.x || 100} y={20} textAnchor="middle" fontSize="13" fontWeight="bold" fill="#333">
        Input (4×4 sample)
      </text>
      <text x={hiddenPos[0]?.x || 600} y={20} textAnchor="middle" fontSize="13" fontWeight="bold" fill="#333">
        Hidden ({hiddenNodes})
      </text>
      <text x={outputPos[0]?.x || 1100} y={20} textAnchor="middle" fontSize="13" fontWeight="bold" fill="#333">
        Output (0-9)
      </text>
    </svg>
  )
}

function calculatePositions(numNodes, height, x) {
  const positions = []
  const spacing = (height - 80) / (numNodes + 1)
  for (let i = 0; i < numNodes; i++) {
    positions.push({ x, y: 40 + spacing * (i + 1) })
  }
  return positions
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 128, g: 128, b: 128 }
}

export default NetworkVisualization
