/**
 * Tests for Login page component.
 *
 * This module tests:
 * - Login form rendering
 * - Form input handling
 * - Form validation
 * - API integration
 * - Loading states
 * - Error handling
 * - Success flow
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { LoginPage } from '../login'
import * as authStore from '@/stores/auth-store'

// Mock dependencies
vi.mock('@/stores/auth-store', () => ({
  default: vi.fn(() => ({
    login: vi.fn(),
    setLoading: vi.fn(),
    isLoading: false,
  })),
}))

vi.mock('sonner', () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}))

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
}))

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    global.fetch = vi.fn()
  })

  describe('Rendering', () => {
    it('renders login form', () => {
      render(<LoginPage />)

      expect(screen.getByLabelText(/username/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
    })

    it('renders POLYMORPH-4 Lite title', () => {
      render(<LoginPage />)

      expect(screen.getByText(/POLYMORPH-4 Lite/i)).toBeInTheDocument()
    })

    it('renders demo credentials', () => {
      render(<LoginPage />)

      expect(screen.getByText(/Demo Credentials/i)).toBeInTheDocument()
      expect(screen.getByText(/admin/i)).toBeInTheDocument()
    })
  })

  describe('Form Interaction', () => {
    it('updates username input value', () => {
      render(<LoginPage />)

      const usernameInput = screen.getByLabelText(/username/i) as HTMLInputElement

      fireEvent.change(usernameInput, { target: { value: 'testuser' } })

      expect(usernameInput.value).toBe('testuser')
    })

    it('updates password input value', () => {
      render(<LoginPage />)

      const passwordInput = screen.getByLabelText(/password/i) as HTMLInputElement

      fireEvent.change(passwordInput, { target: { value: 'testpass' } })

      expect(passwordInput.value).toBe('testpass')
    })

    it('toggles password visibility', () => {
      render(<LoginPage />)

      const passwordInput = screen.getByLabelText(/password/i) as HTMLInputElement
      const toggleButton = screen.getByRole('button', { name: '' }) // Eye icon button

      // Initially password should be hidden
      expect(passwordInput.type).toBe('password')

      // Click to show password
      fireEvent.click(toggleButton)
      expect(passwordInput.type).toBe('text')

      // Click to hide password again
      fireEvent.click(toggleButton)
      expect(passwordInput.type).toBe('password')
    })
  })

  describe('Form Validation', () => {
    it('shows error when username is empty', async () => {
      const { toast } = await import('sonner')
      render(<LoginPage />)

      const submitButton = screen.getByRole('button', { name: /sign in/i })

      fireEvent.click(submitButton)

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Please enter both username and password')
      })
    })

    it('shows error when password is empty', async () => {
      const { toast } = await import('sonner')
      render(<LoginPage />)

      const usernameInput = screen.getByLabelText(/username/i)
      fireEvent.change(usernameInput, { target: { value: 'testuser' } })

      const submitButton = screen.getByRole('button', { name: /sign in/i })
      fireEvent.click(submitButton)

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith('Please enter both username and password')
      })
    })
  })

  describe('API Integration', () => {
    it('calls login API with correct credentials', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          user: {
            id: '1',
            username: 'testuser',
            email: 'testuser@polymorph.com',
            role: 'operator',
            isActive: true,
          },
          access_token: 'mock-token',
        }),
      })
      global.fetch = mockFetch

      render(<LoginPage />)

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      fireEvent.change(usernameInput, { target: { value: 'testuser' } })
      fireEvent.change(passwordInput, { target: { value: 'testpass' } })
      fireEvent.click(submitButton)

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8000/auth/login',
          expect.objectContaining({
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              email: 'testuser@polymorph.com',
              password: 'testpass',
            }),
          })
        )
      })
    })

    it('handles successful login', async () => {
      const mockLogin = vi.fn()
      const mockUseAuthStore = authStore.default as any
      mockUseAuthStore.mockReturnValue({
        login: mockLogin,
        setLoading: vi.fn(),
        isLoading: false,
      })

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          user: {
            id: '1',
            username: 'testuser',
            email: 'testuser@polymorph.com',
            role: 'operator',
            isActive: true,
          },
          access_token: 'mock-token',
        }),
      })
      global.fetch = mockFetch

      const { toast } = await import('sonner')

      render(<LoginPage />)

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      fireEvent.change(usernameInput, { target: { value: 'testuser' } })
      fireEvent.change(passwordInput, { target: { value: 'testpass' } })
      fireEvent.click(submitButton)

      await waitFor(() => {
        expect(mockLogin).toHaveBeenCalled()
        expect(toast.success).toHaveBeenCalledWith(expect.stringContaining('Welcome back'))
      })
    })

    it('handles login failure', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 401,
      })
      global.fetch = mockFetch

      const { toast } = await import('sonner')

      render(<LoginPage />)

      const usernameInput = screen.getByLabelText(/username/i)
      const passwordInput = screen.getByLabelText(/password/i)
      const submitButton = screen.getByRole('button', { name: /sign in/i })

      fireEvent.change(usernameInput, { target: { value: 'wronguser' } })
      fireEvent.change(passwordInput, { target: { value: 'wrongpass' } })
      fireEvent.click(submitButton)

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(expect.stringContaining('Login failed'))
      })
    })
  })

  describe('Loading States', () => {
    it('shows loading state during login', async () => {
      const mockUseAuthStore = authStore.default as any
      mockUseAuthStore.mockReturnValue({
        login: vi.fn(),
        setLoading: vi.fn(),
        isLoading: true,
      })

      render(<LoginPage />)

      expect(screen.getByText(/signing in/i)).toBeInTheDocument()
    })

    it('disables inputs during loading', async () => {
      const mockUseAuthStore = authStore.default as any
      mockUseAuthStore.mockReturnValue({
        login: vi.fn(),
        setLoading: vi.fn(),
        isLoading: true,
      })

      render(<LoginPage />)

      const usernameInput = screen.getByLabelText(/username/i) as HTMLInputElement
      const passwordInput = screen.getByLabelText(/password/i) as HTMLInputElement
      const submitButton = screen.getByRole('button', { name: /signing in/i }) as HTMLButtonElement

      expect(usernameInput.disabled).toBe(true)
      expect(passwordInput.disabled).toBe(true)
      expect(submitButton.disabled).toBe(true)
    })
  })
})
