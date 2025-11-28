import apiClient from './client';
import type { WorkflowExecution } from '../types/api';

export const workflowsApi = {
    list: async (): Promise<WorkflowExecution[]> => {
        const response = await apiClient.get('/api/workflow-builder/executions');
        return response.data;
    },

    get: async (runId: string): Promise<WorkflowExecution> => {
        const response = await apiClient.get(`/api/workflow-builder/executions/${runId}`);
        return response.data;
    },
};
