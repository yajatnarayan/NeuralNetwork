const API_URL = 'http://localhost:8000'

console.log('[API CLIENT] Initialized with API_URL:', API_URL)

export async function getWeights() {
  console.log('[API CLIENT] getWeights() called')
  console.log('[API CLIENT] Making POST request to:', API_URL)
  console.log('[API CLIENT] Request body:', JSON.stringify({ getWeights: true }))

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ getWeights: true })
    })

    console.log('[API CLIENT] getWeights response status:', response.status)
    console.log('[API CLIENT] getWeights response ok:', response.ok)

    if (!response.ok) {
      console.error('[API CLIENT] getWeights HTTP error:', response.status, response.statusText)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    console.log('[API CLIENT] getWeights parsed response:', {
      theta1Shape: data.theta1?.length ? `[${data.theta1.length}x${data.theta1[0]?.length}]` : 'missing',
      theta2Shape: data.theta2?.length ? `[${data.theta2.length}x${data.theta2[0]?.length}]` : 'missing',
      hiddenNodes: data.hiddenNodes
    })

    return {
      theta1: data.theta1,
      theta2: data.theta2,
      hiddenNodes: data.hiddenNodes
    }
  } catch (error) {
    console.error('[API CLIENT] getWeights FAILED:', error.message)
    console.error('[API CLIENT] Error stack:', error.stack)
    throw error
  }
}

export async function trainNetwork(trainArray) {
  console.log('[API CLIENT] trainNetwork() called')
  console.log('[API CLIENT] Training data:', {
    samples: trainArray?.length,
    firstSample: trainArray?.[0] ? {
      label: trainArray[0].label,
      inputLength: trainArray[0].y0?.length,
      nonZeroPixels: trainArray[0].y0?.filter(v => v > 0).length
    } : 'none'
  })

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        train: true,
        trainArray
      })
    })

    console.log('[API CLIENT] trainNetwork response status:', response.status)

    if (!response.ok) {
      console.error('[API CLIENT] trainNetwork HTTP error:', response.status, response.statusText)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()
    console.log('[API CLIENT] trainNetwork result:', {
      hasWeights: !!result.weights,
      message: result.message || 'none'
    })
    return result
  } catch (error) {
    console.error('[API CLIENT] trainNetwork FAILED:', error.message)
    throw error
  }
}

export async function testNetwork(image) {
  console.log('[API CLIENT] testNetwork() called')
  console.log('[API CLIENT] Test image:', {
    length: image?.length,
    nonZeroPixels: image?.filter(v => v > 0).length,
    min: image ? Math.min(...image) : null,
    max: image ? Math.max(...image) : null
  })

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        predict: true,
        image
      })
    })

    console.log('[API CLIENT] testNetwork response status:', response.status)

    if (!response.ok) {
      console.error('[API CLIENT] testNetwork HTTP error:', response.status, response.statusText)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()
    console.log('[API CLIENT] testNetwork result:', {
      prediction: result.result,
      hasActivations: !!result.activations,
      outputActivations: result.activations?.output
    })
    return result
  } catch (error) {
    console.error('[API CLIENT] testNetwork FAILED:', error.message)
    throw error
  }
}

export async function getMetrics() {
  console.log('[API CLIENT] getMetrics() called')

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ getMetrics: true })
    })

    console.log('[API CLIENT] getMetrics response status:', response.status)

    if (!response.ok) {
      console.error('[API CLIENT] getMetrics HTTP error:', response.status)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()
    console.log('[API CLIENT] getMetrics result:', result)
    return result
  } catch (error) {
    console.error('[API CLIENT] getMetrics FAILED:', error.message)
    throw error
  }
}

export async function getArchitecture() {
  console.log('[API CLIENT] getArchitecture() called')

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ getArchitecture: true })
    })

    console.log('[API CLIENT] getArchitecture response status:', response.status)

    if (!response.ok) {
      console.error('[API CLIENT] getArchitecture HTTP error:', response.status)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()
    console.log('[API CLIENT] getArchitecture result:', {
      hasArchitecture: !!result.architecture,
      layers: result.architecture?.architecture ? Object.keys(result.architecture.architecture) : []
    })
    return result
  } catch (error) {
    console.error('[API CLIENT] getArchitecture FAILED:', error.message)
    throw error
  }
}

export async function evaluateMnist(sampleSize = 1000) {
  console.log('[API CLIENT] evaluateMnist() called with sampleSize:', sampleSize)

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        evaluateMnist: true,
        sampleSize
      })
    })

    console.log('[API CLIENT] evaluateMnist response status:', response.status)

    if (!response.ok) {
      console.error('[API CLIENT] evaluateMnist HTTP error:', response.status)
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()
    console.log('[API CLIENT] evaluateMnist result:', {
      accuracy: result.result?.accuracy_percent,
      correct: result.result?.correct,
      sampleSize: result.result?.sample_size
    })
    return result
  } catch (error) {
    console.error('[API CLIENT] evaluateMnist FAILED:', error.message)
    throw error
  }
}
