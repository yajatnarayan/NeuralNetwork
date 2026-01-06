import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import ConfidenceChart from './ConfidenceChart'

describe('ConfidenceChart', () => {
  beforeEach(() => {
    console.log('[TEST] Setting up ConfidenceChart tests')
  })

  it('renders SVG element', () => {
    console.log('[TEST] Running: renders SVG element')

    render(<ConfidenceChart activations={null} prediction={null} />)

    const svg = document.querySelector('svg')
    expect(svg).toBeTruthy()
    console.log('[TEST] SVG element found')
  })

  it('shows placeholder text when no activations', () => {
    console.log('[TEST] Running: placeholder text test')

    render(<ConfidenceChart activations={null} prediction={null} />)

    expect(screen.getByText('Draw and test a digit to see confidence')).toBeTruthy()
    console.log('[TEST] Placeholder text displayed')
  })

  it('shows placeholder when activations array is wrong length', () => {
    console.log('[TEST] Running: wrong length activations test')

    render(<ConfidenceChart activations={[0.1, 0.2, 0.3]} prediction={0} />)

    expect(screen.getByText('Draw and test a digit to see confidence')).toBeTruthy()
    console.log('[TEST] Placeholder shown for incorrect length')
  })

  it('renders bars when valid activations provided', () => {
    console.log('[TEST] Running: valid activations test')

    const activations = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]

    render(<ConfidenceChart activations={activations} prediction={7} />)

    // Should render 10 digit labels (0-9)
    const svg = document.querySelector('svg')
    expect(svg).toBeTruthy()

    // Check for digit labels
    for (let i = 0; i <= 9; i++) {
      expect(screen.getByText(String(i))).toBeTruthy()
    }
    console.log('[TEST] All digit labels rendered')
  })

  it('displays confidence percentages', () => {
    console.log('[TEST] Running: confidence percentages test')

    const activations = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]

    render(<ConfidenceChart activations={activations} prediction={7} />)

    // Check for confidence percentage text (80.0%)
    expect(screen.getByText('80.0%')).toBeTruthy()
    console.log('[TEST] Confidence percentage displayed')
  })

  it('highlights predicted digit', () => {
    console.log('[TEST] Running: highlighted prediction test')

    const activations = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.6, 0.05, 0.05]

    render(<ConfidenceChart activations={activations} prediction={7} />)

    // The SVG should contain rects for the bars
    const rects = document.querySelectorAll('rect')
    expect(rects.length).toBeGreaterThan(0)
    console.log('[TEST] Found', rects.length, 'rect elements')
  })

  it('handles zero activations', () => {
    console.log('[TEST] Running: zero activations test')

    const activations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    render(<ConfidenceChart activations={activations} prediction={0} />)

    const svg = document.querySelector('svg')
    expect(svg).toBeTruthy()
    console.log('[TEST] Component renders with zero activations')
  })

  it('handles all equal activations', () => {
    console.log('[TEST] Running: equal activations test')

    const activations = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    render(<ConfidenceChart activations={activations} prediction={5} />)

    const svg = document.querySelector('svg')
    expect(svg).toBeTruthy()
    console.log('[TEST] Component renders with equal activations')
  })

  it('has correct SVG dimensions', () => {
    console.log('[TEST] Running: SVG dimensions test')

    const activations = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]

    render(<ConfidenceChart activations={activations} prediction={7} />)

    const svg = document.querySelector('svg')
    expect(svg.getAttribute('width')).toBe('350')
    expect(svg.getAttribute('height')).toBe('220')
    console.log('[TEST] SVG dimensions are correct')
  })
})
