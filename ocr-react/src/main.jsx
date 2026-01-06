import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

console.log('[Main] React application starting...')
console.log('[Main] Environment:', {
  mode: import.meta.env.MODE,
  dev: import.meta.env.DEV,
  prod: import.meta.env.PROD
})

const rootElement = document.getElementById('root')
console.log('[Main] Root element found:', !!rootElement)

if (!rootElement) {
  console.error('[Main] FATAL: Root element not found!')
  throw new Error('Root element not found')
}

console.log('[Main] Creating React root...')
const root = createRoot(rootElement)

console.log('[Main] Rendering App component...')
root.render(
  <StrictMode>
    <App />
  </StrictMode>,
)

console.log('[Main] Initial render triggered')
