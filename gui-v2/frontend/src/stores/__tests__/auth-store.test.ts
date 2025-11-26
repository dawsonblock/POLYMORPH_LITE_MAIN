/**
 * Tests for authentication store.
 *
 * This module tests:
 * - Store initialization
 * - Login functionality
 * - Logout functionality
 * - Loading state management
 * - User updates
 * - State persistence
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import useAuthStore from '../auth-store'
import type { User } from '@/types'

describe('useAuthStore', () => {
  beforeEach(() => {
    // Reset store before each test
    const { result } = renderHook(() => useAuthStore())
    act(() => {
      result.current.logout()
    })
  })

  describe('Initial State', () => {
    it('initializes with null user', () => {
      const { result } = renderHook(() => useAuthStore())

      expect(result.current.user).toBeNull()
    })

    it('initializes with null token', () => {
      const { result } = renderHook(() => useAuthStore())

      expect(result.current.token).toBeNull()
    })

    it('initializes as not authenticated', () => {
      const { result } = renderHook(() => useAuthStore())

      expect(result.current.isAuthenticated).toBe(false)
    })

    it('initializes with loading false', () => {
      const { result } = renderHook(() => useAuthStore())

      expect(result.current.isLoading).toBe(false)
    })
  })

  describe('Login', () => {
    const mockUser: User = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'operator',
      isActive: true,
    }

    const mockToken = 'mock-jwt-token'

    it('sets user on login', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.login(mockUser, mockToken)
      })

      expect(result.current.user).toEqual(mockUser)
    })

    it('sets token on login', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.login(mockUser, mockToken)
      })

      expect(result.current.token).toBe(mockToken)
    })

    it('sets isAuthenticated to true on login', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.login(mockUser, mockToken)
      })

      expect(result.current.isAuthenticated).toBe(true)
    })

    it('sets isLoading to false on login', () => {
      const { result } = renderHook(() => useAuthStore())

      // Set loading to true first
      act(() => {
        result.current.setLoading(true)
      })

      // Login should set it to false
      act(() => {
        result.current.login(mockUser, mockToken)
      })

      expect(result.current.isLoading).toBe(false)
    })
  })

  describe('Logout', () => {
    const mockUser: User = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'operator',
      isActive: true,
    }

    it('clears user on logout', () => {
      const { result } = renderHook(() => useAuthStore())

      // Login first
      act(() => {
        result.current.login(mockUser, 'token')
      })

      // Then logout
      act(() => {
        result.current.logout()
      })

      expect(result.current.user).toBeNull()
    })

    it('clears token on logout', () => {
      const { result } = renderHook(() => useAuthStore())

      // Login first
      act(() => {
        result.current.login(mockUser, 'token')
      })

      // Then logout
      act(() => {
        result.current.logout()
      })

      expect(result.current.token).toBeNull()
    })

    it('sets isAuthenticated to false on logout', () => {
      const { result } = renderHook(() => useAuthStore())

      // Login first
      act(() => {
        result.current.login(mockUser, 'token')
      })

      // Then logout
      act(() => {
        result.current.logout()
      })

      expect(result.current.isAuthenticated).toBe(false)
    })

    it('sets isLoading to false on logout', () => {
      const { result } = renderHook(() => useAuthStore())

      // Login and set loading
      act(() => {
        result.current.login(mockUser, 'token')
        result.current.setLoading(true)
      })

      // Then logout
      act(() => {
        result.current.logout()
      })

      expect(result.current.isLoading).toBe(false)
    })
  })

  describe('Loading State', () => {
    it('can set loading to true', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.setLoading(true)
      })

      expect(result.current.isLoading).toBe(true)
    })

    it('can set loading to false', () => {
      const { result } = renderHook(() => useAuthStore())

      // Set to true first
      act(() => {
        result.current.setLoading(true)
      })

      // Then set to false
      act(() => {
        result.current.setLoading(false)
      })

      expect(result.current.isLoading).toBe(false)
    })

    it('can toggle loading state multiple times', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.setLoading(true)
      })
      expect(result.current.isLoading).toBe(true)

      act(() => {
        result.current.setLoading(false)
      })
      expect(result.current.isLoading).toBe(false)

      act(() => {
        result.current.setLoading(true)
      })
      expect(result.current.isLoading).toBe(true)
    })
  })

  describe('Update User', () => {
    const initialUser: User = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'operator',
      isActive: true,
    }

    beforeEach(() => {
      const { result } = renderHook(() => useAuthStore())
      act(() => {
        result.current.login(initialUser, 'token')
      })
    })

    it('updates user username', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.updateUser({ username: 'newusername' })
      })

      expect(result.current.user?.username).toBe('newusername')
    })

    it('updates user email', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.updateUser({ email: 'new@example.com' })
      })

      expect(result.current.user?.email).toBe('new@example.com')
    })

    it('updates user role', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.updateUser({ role: 'admin' })
      })

      expect(result.current.user?.role).toBe('admin')
    })

    it('preserves other user properties when updating', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.updateUser({ username: 'newusername' })
      })

      expect(result.current.user?.id).toBe('1')
      expect(result.current.user?.email).toBe('test@example.com')
      expect(result.current.user?.role).toBe('operator')
    })

    it('can update multiple properties at once', () => {
      const { result } = renderHook(() => useAuthStore())

      act(() => {
        result.current.updateUser({
          username: 'newusername',
          email: 'new@example.com',
          role: 'admin',
        })
      })

      expect(result.current.user?.username).toBe('newusername')
      expect(result.current.user?.email).toBe('new@example.com')
      expect(result.current.user?.role).toBe('admin')
    })

    it('does nothing when user is null', () => {
      const { result } = renderHook(() => useAuthStore())

      // Logout first
      act(() => {
        result.current.logout()
      })

      // Try to update
      act(() => {
        result.current.updateUser({ username: 'newusername' })
      })

      expect(result.current.user).toBeNull()
    })
  })

  describe('Authentication Flow', () => {
    const mockUser: User = {
      id: '1',
      username: 'testuser',
      email: 'test@example.com',
      role: 'operator',
      isActive: true,
    }

    it('completes full login-logout cycle', () => {
      const { result } = renderHook(() => useAuthStore())

      // Start unauthenticated
      expect(result.current.isAuthenticated).toBe(false)

      // Login
      act(() => {
        result.current.login(mockUser, 'token')
      })
      expect(result.current.isAuthenticated).toBe(true)
      expect(result.current.user).toEqual(mockUser)

      // Logout
      act(() => {
        result.current.logout()
      })
      expect(result.current.isAuthenticated).toBe(false)
      expect(result.current.user).toBeNull()
    })

    it('handles login with loading state', () => {
      const { result } = renderHook(() => useAuthStore())

      // Start loading
      act(() => {
        result.current.setLoading(true)
      })
      expect(result.current.isLoading).toBe(true)

      // Login (should clear loading)
      act(() => {
        result.current.login(mockUser, 'token')
      })
      expect(result.current.isLoading).toBe(false)
      expect(result.current.isAuthenticated).toBe(true)
    })
  })
})
