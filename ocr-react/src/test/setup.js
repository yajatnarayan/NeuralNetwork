import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock canvas for tests
class MockCanvas {
  constructor() {
    this.ctx = {
      fillStyle: '',
      strokeStyle: '',
      fillRect: () => {},
      strokeRect: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      stroke: () => {},
      clearRect: () => {},
    }
  }
  getContext() {
    return this.ctx
  }
}

// Mock HTMLCanvasElement
HTMLCanvasElement.prototype.getContext = function() {
  return {
    fillStyle: '',
    strokeStyle: '',
    fillRect: () => {},
    strokeRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {},
    clearRect: () => {},
    canvas: this,
  }
}

// Mock getBoundingClientRect
Element.prototype.getBoundingClientRect = function() {
  return {
    width: 200,
    height: 200,
    top: 0,
    left: 0,
    right: 200,
    bottom: 200,
    x: 0,
    y: 0,
    toJSON: () => {}
  }
}

// Global fetch mock setup
globalThis.fetch = vi.fn()

console.log('[TEST SETUP] Test environment initialized')
