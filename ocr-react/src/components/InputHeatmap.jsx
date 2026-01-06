import { useRef, useEffect } from 'react'
import './InputHeatmap.css'

const SIZE = 200
const GRID_SIZE = 20
const CELL_SIZE = SIZE / GRID_SIZE

console.log('[InputHeatmap] Module loaded with config:', { SIZE, GRID_SIZE, CELL_SIZE })

function InputHeatmap({ data }) {
  console.log('[InputHeatmap] Rendering with data:', {
    hasData: !!data,
    dataLength: data?.length,
    dataType: typeof data,
    isArray: Array.isArray(data),
    nonZeroValues: data ? data.filter(v => v > 0).length : 0
  })

  const canvasRef = useRef(null)

  useEffect(() => {
    console.log('[InputHeatmap] useEffect triggered')
    const canvas = canvasRef.current
    if (!canvas) {
      console.error('[InputHeatmap] Canvas ref is null!')
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      console.error('[InputHeatmap] Could not get 2D context!')
      return
    }

    // Clear to black
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, SIZE, SIZE)

    // Draw heatmap if data available
    if (data && data.length === 400) {
      console.log('[InputHeatmap] Drawing heatmap with', data.length, 'values')
      let minVal = Infinity, maxVal = -Infinity
      for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
          const idx = i * GRID_SIZE + j
          const value = data[idx]
          minVal = Math.min(minVal, value)
          maxVal = Math.max(maxVal, value)
          const brightness = Math.floor(value * 255)
          ctx.fillStyle = `rgb(${brightness},${brightness},${brightness})`
          ctx.fillRect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        }
      }
      console.log('[InputHeatmap] Heatmap drawn, value range:', { minVal, maxVal })
    } else {
      console.log('[InputHeatmap] No valid data to display, expected 400 values, got:', data?.length)
    }
  }, [data])

  return (
    <canvas
      ref={canvasRef}
      width={SIZE}
      height={SIZE}
      className="input-heatmap"
    />
  )
}

export default InputHeatmap
