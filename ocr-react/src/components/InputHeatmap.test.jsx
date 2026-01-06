import { describe, it, expect, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import InputHeatmap from './InputHeatmap'

describe('InputHeatmap', () => {
  beforeEach(() => {
    console.log('[TEST] Setting up InputHeatmap tests')
  })

  it('renders canvas element', () => {
    console.log('[TEST] Running: renders canvas element')

    render(<InputHeatmap data={null} />)

    const canvas = document.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Canvas element found')
  })

  it('renders with correct dimensions', () => {
    console.log('[TEST] Running: canvas dimensions test')

    render(<InputHeatmap data={null} />)

    const canvas = document.querySelector('canvas')
    expect(canvas.width).toBe(200)
    expect(canvas.height).toBe(200)
    console.log('[TEST] Canvas dimensions:', { width: canvas.width, height: canvas.height })
  })

  it('handles null data gracefully', () => {
    console.log('[TEST] Running: null data test')

    const { container } = render(<InputHeatmap data={null} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with null data')
  })

  it('handles undefined data gracefully', () => {
    console.log('[TEST] Running: undefined data test')

    const { container } = render(<InputHeatmap data={undefined} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with undefined data')
  })

  it('handles empty array gracefully', () => {
    console.log('[TEST] Running: empty array test')

    const { container } = render(<InputHeatmap data={[]} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with empty array')
  })

  it('handles incorrect data length gracefully', () => {
    console.log('[TEST] Running: incorrect data length test')

    const incorrectData = Array(100).fill(0.5) // Should be 400
    const { container } = render(<InputHeatmap data={incorrectData} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with incorrect length data')
  })

  it('accepts valid 400-element array', () => {
    console.log('[TEST] Running: valid data test')

    const validData = Array(400).fill(0).map((_, i) => i / 400)
    const { container } = render(<InputHeatmap data={validData} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with valid data')
  })

  it('handles data with all zeros', () => {
    console.log('[TEST] Running: all zeros test')

    const zeroData = Array(400).fill(0)
    const { container } = render(<InputHeatmap data={zeroData} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with all zeros')
  })

  it('handles data with all ones', () => {
    console.log('[TEST] Running: all ones test')

    const onesData = Array(400).fill(1)
    const { container } = render(<InputHeatmap data={onesData} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Component renders with all ones')
  })

  it('updates when data changes', () => {
    console.log('[TEST] Running: data update test')

    const { container, rerender } = render(<InputHeatmap data={null} />)

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeTruthy()

    // Update with new data
    const newData = Array(400).fill(0.5)
    rerender(<InputHeatmap data={newData} />)

    expect(canvas).toBeTruthy()
    console.log('[TEST] Component updates when data changes')
  })
})
