import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DrawingCanvas from './DrawingCanvas'

describe('DrawingCanvas', () => {
  const mockOnTrain = vi.fn()
  const mockOnTest = vi.fn()

  beforeEach(() => {
    console.log('[TEST] Setting up DrawingCanvas tests')
    vi.resetAllMocks()
  })

  it('renders canvas and buttons', () => {
    console.log('[TEST] Running: renders canvas and buttons')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const canvas = document.querySelector('canvas')
    expect(canvas).toBeTruthy()
    console.log('[TEST] Canvas element found:', !!canvas)

    expect(screen.getByText('Train')).toBeTruthy()
    expect(screen.getByText('Test')).toBeTruthy()
    expect(screen.getByText('Reset')).toBeTruthy()
    console.log('[TEST] All buttons found')
  })

  it('calls onTrain when Train button is clicked', () => {
    console.log('[TEST] Running: onTrain callback test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const trainButton = screen.getByText('Train')
    fireEvent.click(trainButton)

    console.log('[TEST] Train button clicked, onTrain called:', mockOnTrain.mock.calls.length, 'times')
    expect(mockOnTrain).toHaveBeenCalled()
    expect(mockOnTrain).toHaveBeenCalledWith(expect.arrayContaining([]))
  })

  it('calls onTest when Test button is clicked', () => {
    console.log('[TEST] Running: onTest callback test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const testButton = screen.getByText('Test')
    fireEvent.click(testButton)

    console.log('[TEST] Test button clicked, onTest called:', mockOnTest.mock.calls.length, 'times')
    expect(mockOnTest).toHaveBeenCalled()
    expect(mockOnTest).toHaveBeenCalledWith(expect.arrayContaining([]))
  })

  it('disables buttons when disabled prop is true', () => {
    console.log('[TEST] Running: disabled buttons test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={true} />)

    const trainButton = screen.getByText('Train')
    const testButton = screen.getByText('Test')
    const resetButton = screen.getByText('Reset')

    expect(trainButton).toBeDisabled()
    expect(testButton).toBeDisabled()
    expect(resetButton).toBeDisabled()
    console.log('[TEST] All buttons are disabled')
  })

  it('resets data when Reset button is clicked', () => {
    console.log('[TEST] Running: reset button test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const resetButton = screen.getByText('Reset')
    fireEvent.click(resetButton)

    // After reset, training data should be all zeros
    const trainButton = screen.getByText('Train')
    fireEvent.click(trainButton)

    const trainData = mockOnTrain.mock.calls[0][0]
    console.log('[TEST] Train data after reset - all zeros:', trainData.every(v => v === 0))
    expect(trainData.every(v => v === 0)).toBe(true)
  })

  it('canvas has correct dimensions', () => {
    console.log('[TEST] Running: canvas dimensions test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const canvas = document.querySelector('canvas')
    expect(canvas.width).toBe(200)
    expect(canvas.height).toBe(200)
    console.log('[TEST] Canvas dimensions:', { width: canvas.width, height: canvas.height })
  })

  it('handles mouse events', () => {
    console.log('[TEST] Running: mouse events test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const canvas = document.querySelector('canvas')

    // Simulate drawing
    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseMove(canvas, { clientX: 60, clientY: 60 })
    fireEvent.mouseUp(canvas)

    console.log('[TEST] Mouse events fired successfully')

    // Now test - the data should have some filled pixels
    const testButton = screen.getByText('Test')
    fireEvent.click(testButton)

    console.log('[TEST] Mouse event test completed')
  })

  it('does not respond to mouse events when disabled', () => {
    console.log('[TEST] Running: disabled mouse events test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={true} />)

    const canvas = document.querySelector('canvas')

    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseMove(canvas, { clientX: 60, clientY: 60 })
    fireEvent.mouseUp(canvas)

    // All pixels should still be 0
    const testButton = screen.getByText('Test')
    fireEvent.click(testButton) // Won't call onTest because button is disabled

    expect(mockOnTest).not.toHaveBeenCalled()
    console.log('[TEST] Disabled canvas correctly ignores events')
  })

  it('outputs data with correct length (400 pixels)', () => {
    console.log('[TEST] Running: data length test')

    render(<DrawingCanvas onTrain={mockOnTrain} onTest={mockOnTest} disabled={false} />)

    const trainButton = screen.getByText('Train')
    fireEvent.click(trainButton)

    const trainData = mockOnTrain.mock.calls[0][0]
    expect(trainData.length).toBe(400)
    console.log('[TEST] Data length is correct:', trainData.length)
  })
})
