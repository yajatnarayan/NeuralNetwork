import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import MetricsPanel from './MetricsPanel'

// Mock the API client
vi.mock('../api/client', () => ({
  getArchitecture: vi.fn(),
  evaluateMnist: vi.fn()
}))

import { getArchitecture, evaluateMnist } from '../api/client'

describe('MetricsPanel', () => {
  const mockArchitecture = {
    architecture: {
      architecture: {
        input_layer: { size: 400 },
        hidden_layer: { size: 25 },
        output_layer: { size: 10 }
      },
      parameters: {
        theta1: { shape: [25, 400], count: 10000 },
        b1: { shape: [25], count: 25 },
        theta2: { shape: [10, 25], count: 250 },
        b2: { shape: [10], count: 10 },
        total: 10285
      }
    }
  }

  const mockMnistResult = {
    result: {
      accuracy_percent: 92.5,
      correct: 925,
      sample_size: 1000,
      per_digit_accuracy: {
        '0': 0.95, '1': 0.98, '2': 0.90, '3': 0.88, '4': 0.92,
        '5': 0.89, '6': 0.94, '7': 0.91, '8': 0.87, '9': 0.93
      }
    },
    architecture: mockArchitecture.architecture
  }

  beforeEach(() => {
    console.log('[TEST] Setting up MetricsPanel tests')
    vi.resetAllMocks()
    getArchitecture.mockResolvedValue(mockArchitecture)
    evaluateMnist.mockResolvedValue(mockMnistResult)
  })

  afterEach(() => {
    console.log('[TEST] Cleaning up MetricsPanel tests')
  })

  it('renders loading state initially', () => {
    console.log('[TEST] Running: loading state test')

    getArchitecture.mockImplementation(() => new Promise(() => {})) // Never resolves

    render(<MetricsPanel />)

    expect(screen.getByText('Loading architecture...')).toBeTruthy()
    console.log('[TEST] Loading state displayed')
  })

  it('loads architecture on mount', async () => {
    console.log('[TEST] Running: load architecture test')

    render(<MetricsPanel />)

    await waitFor(() => {
      expect(getArchitecture).toHaveBeenCalled()
    })

    console.log('[TEST] getArchitecture was called on mount')
  })

  it('displays architecture when loaded', async () => {
    console.log('[TEST] Running: display architecture test')

    render(<MetricsPanel />)

    await waitFor(() => {
      // Wait for loading to complete - check for layer labels
      expect(screen.getByText('Input')).toBeTruthy()
    })

    expect(screen.getByText('Hidden')).toBeTruthy()
    expect(screen.getByText('Output')).toBeTruthy()
    console.log('[TEST] Architecture labels displayed')
  })

  it('renders MNIST evaluation section', () => {
    console.log('[TEST] Running: MNIST section test')

    render(<MetricsPanel />)

    expect(screen.getByText('MNIST Test Accuracy')).toBeTruthy()
    expect(screen.getByText('Run MNIST Evaluation')).toBeTruthy()
    console.log('[TEST] MNIST evaluation section rendered')
  })

  it('renders sample size selector', () => {
    console.log('[TEST] Running: sample size selector test')

    render(<MetricsPanel />)

    const select = screen.getByRole('combobox')
    expect(select).toBeTruthy()
    console.log('[TEST] Sample size selector found')
  })

  it('calls evaluateMnist when button clicked', async () => {
    console.log('[TEST] Running: evaluate MNIST test')

    render(<MetricsPanel />)

    const button = screen.getByText('Run MNIST Evaluation')
    fireEvent.click(button)

    await waitFor(() => {
      expect(evaluateMnist).toHaveBeenCalledWith(1000) // Default sample size
    })

    console.log('[TEST] evaluateMnist was called')
  })

  it('displays MNIST results after evaluation', async () => {
    console.log('[TEST] Running: MNIST results display test')

    render(<MetricsPanel />)

    const button = screen.getByText('Run MNIST Evaluation')
    fireEvent.click(button)

    await waitFor(() => {
      expect(screen.getByText('92.5%')).toBeTruthy()
    })

    console.log('[TEST] MNIST accuracy displayed')
  })

  it('shows evaluating state during MNIST evaluation', async () => {
    console.log('[TEST] Running: evaluating state test')

    let resolveEvaluate
    evaluateMnist.mockImplementation(() => new Promise((resolve) => {
      resolveEvaluate = () => resolve(mockMnistResult)
    }))

    render(<MetricsPanel />)

    const button = screen.getByText('Run MNIST Evaluation')
    fireEvent.click(button)

    expect(screen.getByText('Evaluating...')).toBeTruthy()
    console.log('[TEST] Evaluating state shown')

    // Resolve the promise
    resolveEvaluate()

    await waitFor(() => {
      expect(screen.getByText('92.5%')).toBeTruthy()
    })
  })

  it('handles architecture loading error', async () => {
    console.log('[TEST] Running: architecture error test')

    getArchitecture.mockRejectedValue(new Error('Network error'))

    render(<MetricsPanel />)

    await waitFor(() => {
      expect(screen.getByText('Failed to load architecture')).toBeTruthy()
    })

    console.log('[TEST] Architecture error displayed')
  })

  it('handles MNIST evaluation error', async () => {
    console.log('[TEST] Running: MNIST error test')

    evaluateMnist.mockRejectedValue(new Error('Evaluation failed'))

    render(<MetricsPanel />)

    const button = screen.getByText('Run MNIST Evaluation')
    fireEvent.click(button)

    await waitFor(() => {
      expect(screen.getByText('MNIST evaluation failed')).toBeTruthy()
    })

    console.log('[TEST] MNIST error displayed')
  })

  it('changes sample size when selector changed', async () => {
    console.log('[TEST] Running: sample size change test')

    render(<MetricsPanel />)

    const select = screen.getByRole('combobox')
    fireEvent.change(select, { target: { value: '5000' } })

    const button = screen.getByText('Run MNIST Evaluation')
    fireEvent.click(button)

    await waitFor(() => {
      expect(evaluateMnist).toHaveBeenCalledWith(5000)
    })

    console.log('[TEST] Sample size changed correctly')
  })

  it('disables button during evaluation', async () => {
    console.log('[TEST] Running: button disabled test')

    let resolveEvaluate
    evaluateMnist.mockImplementation(() => new Promise((resolve) => {
      resolveEvaluate = () => resolve(mockMnistResult)
    }))

    render(<MetricsPanel />)

    const button = screen.getByText('Run MNIST Evaluation')
    fireEvent.click(button)

    const evaluatingButton = screen.getByText('Evaluating...')
    expect(evaluatingButton).toBeDisabled()
    console.log('[TEST] Button disabled during evaluation')

    resolveEvaluate()

    await waitFor(() => {
      expect(screen.getByText('Run MNIST Evaluation')).not.toBeDisabled()
    })
  })
})
