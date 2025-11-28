import apiClient from './client';
import type { Sample } from '../types/api';

export const samplesApi = {
    list: async (): Promise<Sample[]> => {
        const response = await apiClient.get('/api/samples');
        return response.data;
    },

    get: async (id: string): Promise<Sample> => {
        const response = await apiClient.get(`/api/samples/${id}`);
        return response.data;
    },

    create: async (sample: Partial<Sample>): Promise<Sample> => {
        const response = await apiClient.post('/api/samples', sample);
        return response.data;
    },

    update: async (id: string, updates: Partial<Sample>): Promise<Sample> => {
        const response = await apiClient.put(`/api/samples/${id}`, updates);
        return response.data;
    },
};
