import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ErrorBoundary from '../components/ErrorBoundary'

// Mock the API client to avoid actual network requests
vi.mock('../api/client', () => ({
  getLogs: vi.fn().mockResolvedValue([]),
  getMedications: vi.fn().mockResolvedValue([]),
  getMedicationHistory: vi.fn().mockResolvedValue([]),
  generateSnapshot: vi.fn(),
  generateDoctorPacket: vi.fn(),
  generateTimeline: vi.fn(),
  deleteLog: vi.fn(),
  initSession: vi.fn().mockResolvedValue(undefined),
  getSessionError: vi.fn().mockReturnValue(null),
  parseDegradedReasons: vi.fn().mockReturnValue([]),
}))

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div>Hello World</div>
      </ErrorBoundary>
    )
    expect(screen.getByText('Hello World')).toBeInTheDocument()
  })

  it('renders error UI when child throws', () => {
    // Suppress console.error for the intentional throw
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})

    function ThrowingComponent(): JSX.Element {
      throw new Error('Test crash')
    }

    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>
    )

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('Reload App')).toBeInTheDocument()
    spy.mockRestore()
  })
})

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders the app header', async () => {
    const App = (await import('../App')).default
    await act(async () => {
      render(<App />, { wrapper: createWrapper() })
    })
    expect(screen.getByText('SymptomPal')).toBeInTheDocument()
    expect(screen.getByText('Voice-first health journaling for faster clinical handoffs')).toBeInTheDocument()
  })

  it('renders the Replay Demo button', async () => {
    const App = (await import('../App')).default
    await act(async () => {
      render(<App />, { wrapper: createWrapper() })
    })
    expect(screen.getByText('Replay Demo')).toBeInTheDocument()
  })

  it('renders tab navigation', async () => {
    const App = (await import('../App')).default
    await act(async () => {
      render(<App />, { wrapper: createWrapper() })
    })
    expect(screen.getByText('Record')).toBeInTheDocument()
    expect(screen.getByText('Monitor')).toBeInTheDocument()
    expect(screen.getByText('Meds')).toBeInTheDocument()
    expect(screen.getByText('Doctor')).toBeInTheDocument()
  })

  it('shows session error banner when session fails', async () => {
    const { getSessionError } = await import('../api/client')
    vi.mocked(getSessionError).mockReturnValue('Cannot connect to backend: connection refused')

    const App = (await import('../App')).default
    await act(async () => {
      render(<App />, { wrapper: createWrapper() })
    })

    // Wait for session init effect
    await vi.waitFor(() => {
      expect(screen.getByText(/Cannot connect to backend/)).toBeInTheDocument()
    })
  })

  it('renders safety disclaimer', async () => {
    const App = (await import('../App')).default
    await act(async () => {
      render(<App />, { wrapper: createWrapper() })
    })
    expect(screen.getByText(/does not provide medical advice/i)).toBeInTheDocument()
  })
})
