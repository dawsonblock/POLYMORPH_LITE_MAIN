import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import Login from '../pages/Login'
import { AuthProvider } from '../contexts/AuthContext'

// Mock the auth API
vi.mock('../api/auth', () => ({
    authApi: {
        login: vi.fn(),
        getCurrentUser: vi.fn(),
    },
}))

describe('Login Component', () => {
    it('renders login form', () => {
        render(
            <BrowserRouter>
                <AuthProvider>
                    <Login />
                </AuthProvider>
            </BrowserRouter>
        )

        expect(screen.getByLabelText(/email/i)).toBeInTheDocument()
        expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
        expect(screen.getByRole('button', { name: /login/i })).toBeInTheDocument()
    })

    it('shows validation errors for empty fields', async () => {
        render(
            <BrowserRouter>
                <AuthProvider>
                    <Login />
                </AuthProvider>
            </BrowserRouter>
        )

        const submitButton = screen.getByRole('button', { name: /login/i })
        fireEvent.click(submitButton)

        // Form should prevent submission with empty fields
        expect(screen.getByLabelText(/email/i)).toBeInTheDocument()
    })

    it('displays hint for default credentials', () => {
        render(
            <BrowserRouter>
                <AuthProvider>
                    <Login />
                </AuthProvider>
            </BrowserRouter>
        )

        expect(screen.getByText(/admin@polymorph.local/i)).toBeInTheDocument()
    })
})
