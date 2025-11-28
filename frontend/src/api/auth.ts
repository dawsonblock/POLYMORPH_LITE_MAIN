import apiClient from './client';
import type { LoginRequest, LoginResponse } from '../types/api';

export const authApi = {
    login: async (credentials: LoginRequest): Promise<LoginResponse> => {
        const formData = new FormData();
        formData.append('username', credentials.email);
        formData.append('password', credentials.password);

        const response = await apiClient.post<LoginResponse>('/auth/login', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    getCurrentUser: async () => {
        const response = await apiClient.get('/api/users/me');
        return response.data;
    },
};
