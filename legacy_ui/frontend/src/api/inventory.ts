import apiClient from './client';
import type { InventoryItem } from '../types/api';

export const inventoryApi = {
    list: async (): Promise<InventoryItem[]> => {
        const response = await apiClient.get('/api/inventory');
        return response.data;
    },

    getLowStock: async (): Promise<InventoryItem[]> => {
        const response = await apiClient.get('/api/inventory/low-stock');
        return response.data;
    },

    create: async (item: Partial<InventoryItem>): Promise<InventoryItem> => {
        const response = await apiClient.post('/api/inventory', item);
        return response.data;
    },
};
