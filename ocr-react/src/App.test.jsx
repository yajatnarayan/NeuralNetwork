import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import App from './App'

// Mock all API client functions
vi.mock('./api/client', () => ({
  getWeights: vi.fn(),
  trainNetwork: vi.fn(),
  testNetwork: vi.fn()
}))

import { getWeights, trainNetwork, testNetwork } from './api/client'

describe('App', () => {
  const mockWeights = {
    theta1: Array(25).fill(null).map(() => Array(400).fill(0.1)),
    theta2: Array(10).fill(null).map(() => Array(25).fill(0.1)),
    hiddenNodes: 25
  }

  const mockTestResponse = {
    result: 7,
    activations: {
      input: Array(400).fill(0.5),
      hidden: Array(25).fill(0.5),
      output: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]
    }
  }

  const mockTrainResponse = {
    weights: mockWeights,
    message: 'Training complete'
  }

  beforeEach(() => {
    console.log('[TEST] Setting up App tests')
    vi.resetAllMocks()
    getWeights.mockResolvedValue(mockWeights)
    trainNetwork.mockResolvedValue(mockTrainResponse)
    testNetwork.mockResolvedValue(mockTestResponse)

    // Mock window.alert
    vi.spyOn(window, 'alert').mockImplementation(() => {})
  })

  afterEach(() => {
    console.log('[TEST] Cleaning up App tests')
    vi.restoreAllMocks()
  })

  it('renders the app title', () => {
    console.log('[TEST] Running: app title test')

    render(<App />)

    expect(screen.getByText('OCR Neural Network Demo')).toBeTruthy()
    console.log('[TEST] App title found')
  })

  it('loads weights on mount', async () => {
    console.log('[TEST] Running: load weights test')

    render(<App />)

    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    console.log('[TEST] getWeights called on mount')
  })

  it('renders drawing canvas section', () => {
    console.log('[TEST] Running: drawing canvas section test')

    render(<App />)

    expect(screen.getByText('Drawing Canvas (20×20)')).toBeTruthy()
    console.log('[TEST] Drawing canvas section found')
  })

  it('renders input vector section', () => {
    console.log('[TEST] Running: input vector section test')

    render(<App />)

    expect(screen.getByText('Input Vector (400 dims)')).toBeTruthy()
    console.log('[TEST] Input vector section found')
  })

  it('renders prediction confidence section', () => {
    console.log('[TEST] Running: prediction confidence section test')

    render(<App />)

    expect(screen.getByText('Prediction Confidence')).toBeTruthy()
    console.log('[TEST] Prediction confidence section found')
  })

  it('renders network visualization section', () => {
    console.log('[TEST] Running: network visualization section test')

    render(<App />)

    expect(screen.getByText('Neural Network Architecture (Input → Hidden → Output)')).toBeTruthy()
    console.log('[TEST] Network visualization section found')
  })

  it('renders model metrics section', () => {
    console.log('[TEST] Running: model metrics section test')

    render(<App />)

    expect(screen.getByText('Model Metrics')).toBeTruthy()
    console.log('[TEST] Model metrics section found')
  })

  it('renders digit input field', () => {
    console.log('[TEST] Running: digit input test')

    render(<App />)

    const input = screen.getByPlaceholderText('Digit (0-9)')
    expect(input).toBeTruthy()
    console.log('[TEST] Digit input field found')
  })

  it('displays error when weights fail to load', async () => {
    console.log('[TEST] Running: weights load error test')

    getWeights.mockRejectedValue(new Error('Failed to fetch'))

    render(<App />)

    await waitFor(() => {
      expect(screen.getByText('Failed to load network weights')).toBeTruthy()
    })

    console.log('[TEST] Error message displayed')
  })

  it('updates digit input when typed', () => {
    console.log('[TEST] Running: digit input change test')

    render(<App />)

    const input = screen.getByPlaceholderText('Digit (0-9)')
    fireEvent.change(input, { target: { value: '5' } })

    expect(input.value).toBe('5')
    console.log('[TEST] Digit input updated')
  })

  it('shows loading overlay during operations', async () => {
    console.log('[TEST] Running: loading overlay test')

    let resolveTest
    testNetwork.mockImplementation(() => new Promise((resolve) => {
      resolveTest = () => resolve(mockTestResponse)
    }))

    render(<App />)

    // Wait for initial weights load
    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    // Simulate drawing and testing
    const canvas = document.querySelector('canvas')
    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseUp(canvas)

    const testButton = screen.getByText('Test')
    fireEvent.click(testButton)

    // Check for loading overlay
    const spinner = document.querySelector('.spinner')
    expect(spinner).toBeTruthy()
    console.log('[TEST] Loading spinner shown')

    resolveTest()

    await waitFor(() => {
      expect(document.querySelector('.spinner')).toBeFalsy()
    })
  })

  it('trains network when train button clicked with data', async () => {
    console.log('[TEST] Running: train network test')

    render(<App />)

    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    // Enter a digit
    const input = screen.getByPlaceholderText('Digit (0-9)')
    fireEvent.change(input, { target: { value: '5' } })

    // Draw something
    const canvas = document.querySelector('canvas')
    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseUp(canvas)

    // Click train
    const trainButton = screen.getByText('Train')
    fireEvent.click(trainButton)

    await waitFor(() => {
      expect(trainNetwork).toHaveBeenCalled()
    })

    console.log('[TEST] trainNetwork called')
  })

  it('shows alert when training without digit', async () => {
    console.log('[TEST] Running: train without digit test')

    render(<App />)

    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    // Click train without drawing or entering digit
    const trainButtons = screen.getAllByText('Train')
    fireEvent.click(trainButtons[0])

    expect(window.alert).toHaveBeenCalledWith('Please draw a digit and enter its value (0-9)')
    console.log('[TEST] Alert shown for missing digit')
  })

  it('tests network when test button clicked with drawing', async () => {
    console.log('[TEST] Running: test network test')

    render(<App />)

    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    // Draw something
    const canvas = document.querySelector('canvas')
    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseUp(canvas)

    const testButton = screen.getByText('Test')
    fireEvent.click(testButton)

    await waitFor(() => {
      expect(testNetwork).toHaveBeenCalled()
    })

    console.log('[TEST] testNetwork called')
  })

  it('renders legend items', () => {
    console.log('[TEST] Running: legend items test')

    render(<App />)

    expect(screen.getByText('Positive weights')).toBeTruthy()
    expect(screen.getByText('Negative weights')).toBeTruthy()
    expect(screen.getByText('Input layer')).toBeTruthy()
    expect(screen.getByText('Hidden layer')).toBeTruthy()
    expect(screen.getByText('Output layer')).toBeTruthy()
    console.log('[TEST] All legend items found')
  })

  it('handles test error gracefully', async () => {
    console.log('[TEST] Running: test error handling test')

    testNetwork.mockRejectedValue(new Error('Prediction failed'))

    render(<App />)

    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    // Draw and test
    const canvas = document.querySelector('canvas')
    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseUp(canvas)

    const testButton = screen.getByText('Test')
    fireEvent.click(testButton)

    await waitFor(() => {
      expect(screen.getByText('Prediction failed')).toBeTruthy()
    })

    console.log('[TEST] Test error displayed')
  })

  it('handles train error gracefully', async () => {
    console.log('[TEST] Running: train error handling test')

    trainNetwork.mockRejectedValue(new Error('Training error'))

    render(<App />)

    await waitFor(() => {
      expect(getWeights).toHaveBeenCalled()
    })

    // Enter digit and draw
    const input = screen.getByPlaceholderText('Digit (0-9)')
    fireEvent.change(input, { target: { value: '3' } })

    const canvas = document.querySelector('canvas')
    fireEvent.mouseDown(canvas, { clientX: 50, clientY: 50 })
    fireEvent.mouseUp(canvas)

    const trainButton = screen.getByText('Train')
    fireEvent.click(trainButton)

    await waitFor(() => {
      expect(screen.getByText('Training failed')).toBeTruthy()
    })

    console.log('[TEST] Training error displayed')
  })
})
