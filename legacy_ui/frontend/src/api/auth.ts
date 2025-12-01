import apiClient from './client';
import type { LoginRequest, LoginResponse } from '../types/api';

export const authApi = {
    login: async (credentials: LoginRequest): Promise<LoginResponse> => {
        const response = await apiClient.post<LoginResponse>('/auth/login', credentials);
        return response.data;
    },

    getCurrentUser: async () => {
        const response = await apiClient.get('/api/users/me');
        return response.data;
    },
};
