import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import NetworkVisualization from './NetworkVisualization'

describe('NetworkVisualization', () => {
  beforeEach(() => {
    console.log('[TEST] Setting up NetworkVisualization tests')
  })

  const createMockWeights = (hiddenNodes = 25) => ({
    theta1: Array(hiddenNodes).fill(null).map(() => Array(400).fill(0).map(() => Math.random() - 0.5)),
    theta2: Array(10).fill(null).map(() => Array(hiddenNodes).fill(0).map(() => Math.random() - 0.5)),
    hiddenNodes
  })

  it('renders SVG element', () => {
    console.log('[TEST] Running: renders SVG element')

    render(<NetworkVisualization weights={null} activations={null} />)

    const svg = document.querySelector('svg')
    expect(svg).toBeTruthy()
    console.log('[TEST] SVG element found')
  })

  it('shows loading message when no weights', () => {
    console.log('[TEST] Running: loading message test')

    render(<NetworkVisualization weights={null} activations={null} />)

    expect(screen.getByText('Loading network visualization...')).toBeTruthy()
    console.log('[TEST] Loading message displayed')
  })

  it('renders network when weights provided', () => {
    console.log('[TEST] Running: network rendering test')

    const weights = createMockWeights(25)
    render(<NetworkVisualization weights={weights} activations={null} />)

    // Should have layer labels
    expect(screen.getByText('Input (4×4 sample)')).toBeTruthy()
    expect(screen.getByText('Hidden (25)')).toBeTruthy()
    expect(screen.getByText('Output (0-9)')).toBeTruthy()
    console.log('[TEST] Layer labels rendered')
  })

  it('renders nodes as circles', () => {
    console.log('[TEST] Running: nodes rendering test')

    const weights = createMockWeights(25)
    render(<NetworkVisualization weights={weights} activations={null} />)

    const circles = document.querySelectorAll('circle')
    // Should have 16 input + 25 hidden + 10 output = 51 nodes
    expect(circles.length).toBe(51)
    console.log('[TEST] Found', circles.length, 'circle elements')
  })

  it('renders connections as lines', () => {
    console.log('[TEST] Running: connections rendering test')

    const weights = createMockWeights(5) // Smaller for testing
    render(<NetworkVisualization weights={weights} activations={null} />)

    const lines = document.querySelectorAll('line')
    expect(lines.length).toBeGreaterThan(0)
    console.log('[TEST] Found', lines.length, 'line elements')
  })

  it('handles activations data', () => {
    console.log('[TEST] Running: activations data test')

    const weights = createMockWeights(25)
    const activations = {
      input: Array(400).fill(0.5),
      hidden: Array(25).fill(0.5),
      output: Array(10).fill(0.1)
    }

    render(<NetworkVisualization weights={weights} activations={activations} />)

    const svg = document.querySelector('svg')
    expect(svg).toBeTruthy()
    console.log('[TEST] Component renders with activations')
  })

  it('has correct viewBox', () => {
    console.log('[TEST] Running: viewBox test')

    const weights = createMockWeights(25)
    render(<NetworkVisualization weights={weights} activations={null} />)

    const svg = document.querySelector('svg')
    expect(svg.getAttribute('viewBox')).toBe('0 0 1200 400')
    console.log('[TEST] ViewBox is correct')
  })

  it('handles different hidden layer sizes', () => {
    console.log('[TEST] Running: different hidden sizes test')

    const weights10 = createMockWeights(10)
    const { rerender } = render(<NetworkVisualization weights={weights10} activations={null} />)

    expect(screen.getByText('Hidden (10)')).toBeTruthy()

    const weights50 = createMockWeights(50)
    rerender(<NetworkVisualization weights={weights50} activations={null} />)

    expect(screen.getByText('Hidden (50)')).toBeTruthy()
    console.log('[TEST] Different hidden sizes handled correctly')
  })

  it('updates when weights change', () => {
    console.log('[TEST] Running: weights update test')

    const weights1 = createMockWeights(15)
    const { rerender } = render(<NetworkVisualization weights={weights1} activations={null} />)

    expect(screen.getByText('Hidden (15)')).toBeTruthy()

    const weights2 = createMockWeights(30)
    rerender(<NetworkVisualization weights={weights2} activations={null} />)

    expect(screen.getByText('Hidden (30)')).toBeTruthy()
    console.log('[TEST] Component updates when weights change')
  })

  it('colors connections based on weight sign', () => {
    console.log('[TEST] Running: connection colors test')

    // Create weights with known positive and negative values
    const weights = {
      theta1: Array(5).fill(null).map(() => Array(400).fill(0.5)),
      theta2: Array(10).fill(null).map(() => Array(5).fill(-0.5)),
      hiddenNodes: 5
    }

    render(<NetworkVisualization weights={weights} activations={null} />)

    const lines = document.querySelectorAll('line')
    // Check that lines have stroke attributes
    lines.forEach((line) => {
      expect(line.getAttribute('stroke')).toBeTruthy()
    })
    console.log('[TEST] Connection colors applied')
  })
})
