import { useRef, useEffect, useState, useCallback } from 'react'
import './DrawingCanvas.css'

const CANVAS_SIZE = 200
const GRID_SIZE = 20
const PIXEL_SIZE = CANVAS_SIZE / GRID_SIZE

console.log('[DrawingCanvas] Module loaded with config:', { CANVAS_SIZE, GRID_SIZE, PIXEL_SIZE })

function DrawingCanvas({ onTrain, onTest, disabled }) {
  console.log('[DrawingCanvas] Component rendering with props:', { disabled, hasOnTrain: !!onTrain, hasOnTest: !!onTest })

  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [data, setData] = useState(Array(GRID_SIZE * GRID_SIZE).fill(0))

  const drawCanvas = useCallback((ctx, currentData) => {
    console.log('[DrawingCanvas] drawCanvas() called, filledPixels:', currentData.filter(v => v === 1).length)

    // Clear and draw grid
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)

    // Draw grid lines
    ctx.strokeStyle = '#0000FF'
    for (let i = PIXEL_SIZE; i < CANVAS_SIZE; i += PIXEL_SIZE) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, CANVAS_SIZE)
      ctx.stroke()

      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(CANVAS_SIZE, i)
      ctx.stroke()
    }

    // Draw filled pixels
    ctx.fillStyle = '#FFF'
    currentData.forEach((value, index) => {
      if (value === 1) {
        const x = (index % GRID_SIZE) * PIXEL_SIZE
        const y = Math.floor(index / GRID_SIZE) * PIXEL_SIZE
        ctx.fillRect(x, y, PIXEL_SIZE, PIXEL_SIZE)
      }
    })
    console.log('[DrawingCanvas] Canvas drawing complete')
  }, [])

  useEffect(() => {
    console.log('[DrawingCanvas] useEffect triggered - data changed')
    const canvas = canvasRef.current
    if (!canvas) {
      console.error('[DrawingCanvas] Canvas ref is null!')
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      console.error('[DrawingCanvas] Could not get 2D context!')
      return
    }
    console.log('[DrawingCanvas] Got 2D context, drawing canvas')
    drawCanvas(ctx, data)
  }, [data, drawCanvas])

  const fillPixel = (x, y) => {
    const xPixel = Math.floor(x / PIXEL_SIZE)
    const yPixel = Math.floor(y / PIXEL_SIZE)

    if (xPixel < 0 || yPixel < 0 || xPixel >= GRID_SIZE || yPixel >= GRID_SIZE) {
      console.log('[DrawingCanvas] fillPixel out of bounds:', { x, y, xPixel, yPixel })
      return
    }

    const index = yPixel * GRID_SIZE + xPixel
    const newData = [...data]
    newData[index] = 1
    setData(newData)
  }

  const handleMouseDown = (e) => {
    console.log('[DrawingCanvas] Mouse down event, disabled:', disabled)
    if (disabled) return
    setIsDrawing(true)
    const rect = canvasRef.current.getBoundingClientRect()
    console.log('[DrawingCanvas] Canvas rect:', rect)
    fillPixel(e.clientX - rect.left, e.clientY - rect.top)
  }

  const handleMouseMove = (e) => {
    if (!isDrawing || disabled) return
    const rect = canvasRef.current.getBoundingClientRect()
    fillPixel(e.clientX - rect.left, e.clientY - rect.top)
  }

  const handleMouseUp = () => {
    console.log('[DrawingCanvas] Mouse up - stopped drawing')
    setIsDrawing(false)
  }

  const handleReset = () => {
    console.log('[DrawingCanvas] Reset canvas')
    setData(Array(GRID_SIZE * GRID_SIZE).fill(0))
  }

  const handleTrain = () => {
    console.log('[DrawingCanvas] Train button clicked')
    console.log('[DrawingCanvas] Training data:', {
      totalPixels: data.length,
      filledPixels: data.filter(v => v === 1).length,
      dataPreview: data.slice(0, 20)
    })
    onTrain(data)
    handleReset()
  }

  const handleTest = () => {
    console.log('[DrawingCanvas] Test button clicked')
    console.log('[DrawingCanvas] Test data:', {
      totalPixels: data.length,
      filledPixels: data.filter(v => v === 1).length,
      dataPreview: data.slice(0, 20)
    })
    onTest(data)
    handleReset()
  }

  return (
    <div className="drawing-canvas-container">
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="drawing-canvas"
        style={{ cursor: disabled ? 'not-allowed' : 'crosshair' }}
      />
      <div className="button-group">
        <button onClick={handleTrain} disabled={disabled}>Train</button>
        <button onClick={handleTest} disabled={disabled}>Test</button>
        <button onClick={handleReset} disabled={disabled}>Reset</button>
      </div>
    </div>
  )
}

export default DrawingCanvas
