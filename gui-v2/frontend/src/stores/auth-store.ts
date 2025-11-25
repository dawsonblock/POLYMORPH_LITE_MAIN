import { create } from 'zustand'
import { persist, devtools } from 'zustand/middleware'
import type { User } from '@/types'

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean

  // Actions
  login: (user: User, token: string) => void
  logout: () => void
  setLoading: (loading: boolean) => void
  updateUser: (user: Partial<User>) => void
}

const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,

        login: (user, token) =>
          set(
            {
              user,
              token,
              isAuthenticated: true,
              isLoading: false,
            },
            false,
            'auth/login'
          ),

        logout: () =>
          set(
            {
              user: null,
              token: null,
              isAuthenticated: false,
              isLoading: false,
            },
            false,
            'auth/logout'
          ),

        setLoading: (isLoading) =>
          set(
            { isLoading },
            false,
            'auth/setLoading'
          ),

        updateUser: (userData) =>
          set(
            (state) => ({
              user: state.user ? { ...state.user, ...userData } : null,
            }),
            false,
            'auth/updateUser'
          ),
      }),
      {
        name: 'auth-store',
        partialize: (state) => ({
          user: state.user,
          token: state.token,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    {
      name: 'auth-store',
    }
  )
)

export default useAuthStore