import React, { createContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { authApi } from '../api/auth';
import type { LoginRequest, User } from '../types/api';

interface AuthContextType {
    user: User | null;
    loading: boolean;
    login: (credentials: LoginRequest) => Promise<void>;
    logout: () => void;
    isAuthenticated: boolean;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if user is already logged in
        const token = localStorage.getItem('access_token');
        if (token) {
            // TODO: Fetch current user from API
            // For now, just set loading to false
            setLoading(false);
        } else {
            setLoading(false);
        }
    }, []);

    const login = async (credentials: LoginRequest) => {
        const response = await authApi.login(credentials);
        localStorage.setItem('access_token', response.access_token);
        // TODO: Fetch user details
        setUser({ email: credentials.email, role: 'user', permissions: [] });
    };

    const logout = () => {
        localStorage.removeItem('access_token');
        setUser(null);
        window.location.href = '/login';
    };

    const isAuthenticated = !!localStorage.getItem('access_token');

    return (
        <AuthContext.Provider value={{ user, loading, login, logout, isAuthenticated }}>
            {children}
        </AuthContext.Provider>
    );
};
