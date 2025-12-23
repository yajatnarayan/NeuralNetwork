import { useRef, useEffect } from 'react'
import './InputHeatmap.css'

const SIZE = 200
const GRID_SIZE = 20
const CELL_SIZE = SIZE / GRID_SIZE

function InputHeatmap({ data }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')

    // Clear to black
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, SIZE, SIZE)

    // Draw heatmap if data available
    if (data && data.length === 400) {
      for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
          const idx = i * GRID_SIZE + j
          const value = data[idx]
          const brightness = Math.floor(value * 255)
          ctx.fillStyle = `rgb(${brightness},${brightness},${brightness})`
          ctx.fillRect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        }
      }
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
