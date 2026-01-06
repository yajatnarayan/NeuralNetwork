import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { getWeights, trainNetwork, testNetwork, getMetrics, getArchitecture, evaluateMnist } from './client'

describe('API Client', () => {
  beforeEach(() => {
    console.log('[TEST] Setting up API client tests')
    vi.resetAllMocks()
  })

  afterEach(() => {
    console.log('[TEST] Cleaning up API client tests')
  })

  describe('getWeights', () => {
    it('should fetch weights successfully', async () => {
      console.log('[TEST] Running: getWeights success test')

      const mockWeights = {
        theta1: [[0.1, 0.2], [0.3, 0.4]],
        theta2: [[0.5, 0.6]],
        hiddenNodes: 25
      }

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockWeights)
      })

      const result = await getWeights()

      console.log('[TEST] getWeights result:', result)
      expect(result).toEqual(mockWeights)
      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:8000',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ getWeights: true })
        })
      )
    })

    it('should throw error on HTTP failure', async () => {
      console.log('[TEST] Running: getWeights HTTP error test')

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      })

      await expect(getWeights()).rejects.toThrow('HTTP error! status: 500')
      console.log('[TEST] getWeights correctly threw error on HTTP 500')
    })

    it('should throw error on network failure', async () => {
      console.log('[TEST] Running: getWeights network error test')

      globalThis.fetch = vi.fn().mockRejectedValue(new Error('Network error'))

      await expect(getWeights()).rejects.toThrow('Network error')
      console.log('[TEST] getWeights correctly threw network error')
    })
  })

  describe('trainNetwork', () => {
    it('should train network successfully', async () => {
      console.log('[TEST] Running: trainNetwork success test')

      const trainData = [{
        y0: Array(400).fill(0).map((_, i) => i < 100 ? 1 : 0),
        label: 5
      }]

      const mockResponse = {
        weights: {
          theta1: [[0.1]],
          theta2: [[0.2]],
          hiddenNodes: 25
        },
        message: 'Training complete'
      }

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse)
      })

      const result = await trainNetwork(trainData)

      console.log('[TEST] trainNetwork result:', result)
      expect(result).toEqual(mockResponse)
      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:8000',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"train":true')
        })
      )
    })

    it('should handle training failure', async () => {
      console.log('[TEST] Running: trainNetwork failure test')

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request'
      })

      await expect(trainNetwork([{ y0: [], label: 0 }])).rejects.toThrow('HTTP error! status: 400')
    })
  })

  describe('testNetwork', () => {
    it('should test network successfully', async () => {
      console.log('[TEST] Running: testNetwork success test')

      const testImage = Array(400).fill(0)
      testImage[100] = 1
      testImage[101] = 1

      const mockResponse = {
        result: 7,
        activations: {
          input: testImage,
          hidden: Array(25).fill(0.5),
          output: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1]
        }
      }

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse)
      })

      const result = await testNetwork(testImage)

      console.log('[TEST] testNetwork result:', result)
      expect(result.result).toBe(7)
      expect(result.activations).toBeDefined()
      expect(result.activations.output).toHaveLength(10)
    })

    it('should handle prediction failure', async () => {
      console.log('[TEST] Running: testNetwork failure test')

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Server Error'
      })

      await expect(testNetwork(Array(400).fill(0))).rejects.toThrow('HTTP error! status: 500')
    })
  })

  describe('getMetrics', () => {
    it('should fetch metrics successfully', async () => {
      console.log('[TEST] Running: getMetrics success test')

      const mockMetrics = {
        trainingCount: 100,
        accuracy: 0.95
      }

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockMetrics)
      })

      const result = await getMetrics()

      console.log('[TEST] getMetrics result:', result)
      expect(result).toEqual(mockMetrics)
    })
  })

  describe('getArchitecture', () => {
    it('should fetch architecture successfully', async () => {
      console.log('[TEST] Running: getArchitecture success test')

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

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockArchitecture)
      })

      const result = await getArchitecture()

      console.log('[TEST] getArchitecture result:', result)
      expect(result.architecture).toBeDefined()
    })
  })

  describe('evaluateMnist', () => {
    it('should evaluate MNIST successfully', async () => {
      console.log('[TEST] Running: evaluateMnist success test')

      const mockResult = {
        result: {
          accuracy_percent: 92.5,
          correct: 925,
          sample_size: 1000,
          per_digit_accuracy: {
            '0': 0.95, '1': 0.98, '2': 0.90, '3': 0.88, '4': 0.92,
            '5': 0.89, '6': 0.94, '7': 0.91, '8': 0.87, '9': 0.93
          }
        }
      }

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResult)
      })

      const result = await evaluateMnist(1000)

      console.log('[TEST] evaluateMnist result:', result)
      expect(result.result.accuracy_percent).toBe(92.5)
      expect(result.result.sample_size).toBe(1000)
    })

    it('should use default sample size', async () => {
      console.log('[TEST] Running: evaluateMnist default sample size test')

      globalThis.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ result: {} })
      })

      await evaluateMnist()

      const callBody = JSON.parse(globalThis.fetch.mock.calls[0][1].body)
      console.log('[TEST] evaluateMnist called with:', callBody)
      expect(callBody.sampleSize).toBe(1000)
    })
  })
})
