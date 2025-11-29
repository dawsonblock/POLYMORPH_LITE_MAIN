import { apiClient } from './client';

export interface WorkflowNode {
    id: string;
    type: string;
    position: { x: number; y: number };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data: Record<string, any>;
}

export interface WorkflowEdge {
    id: string;
    source: string;
    target: string;
    sourceHandle?: string;
    targetHandle?: string;
    label?: string;
}

export interface WorkflowDefinition {
    id: string;
    workflow_name: string;
    version: number;
    definition: {
        nodes: WorkflowNode[];
        edges: WorkflowEdge[];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        metadata: Record<string, any>;
    };
    definition_hash: string;
    is_active: boolean;
    is_approved: boolean;
    created_by: string;
    created_at: string;
    approved_by?: string;
    approved_at?: string;
}

export interface WorkflowExecutionCreate {
    workflow_name: string;
    workflow_version?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    parameters?: Record<string, any>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    metadata?: Record<string, any>;
}

export interface WorkflowExecution {
    id: string;
    run_id: string;
    workflow_version_id: string;
    started_at: string;
    completed_at?: string;
    status: string;
    operator: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    results: Record<string, any>;
    error_message?: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    run_metadata: Record<string, any>;
}

export const workflowBuilderApi = {
    // List all workflow names
    listWorkflows: async (): Promise<string[]> => {
        const response = await apiClient.get('/api/workflow-builder/workflows');
        return response.data;
    },

    // Get all versions of a workflow
    listWorkflowVersions: async (workflowName: string): Promise<WorkflowDefinition[]> => {
        const response = await apiClient.get(`/api/workflow-builder/workflows/${workflowName}`);
        return response.data;
    },

    // Get specific workflow version
    getWorkflowVersion: async (workflowName: string, version: number): Promise<WorkflowDefinition> => {
        const response = await apiClient.get(`/api/workflow-builder/workflows/${workflowName}/v/${version}`);
        return response.data;
    },

    // Get active workflow version
    getActiveWorkflow: async (workflowName: string): Promise<WorkflowDefinition> => {
        const response = await apiClient.get(`/api/workflow-builder/workflows/${workflowName}/active`);
        return response.data;
    },

    // Create new workflow definition
    createWorkflow: async (data: {
        workflow_name: string;
        nodes: WorkflowNode[];
        edges: WorkflowEdge[];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        metadata?: Record<string, any>;
    }): Promise<WorkflowDefinition> => {
        const response = await apiClient.post('/api/workflow-builder/workflows', data);
        return response.data;
    },

    // Approve workflow version
    approveWorkflow: async (workflowName: string, version: number): Promise<{ message: string }> => {
        const response = await apiClient.post(
            `/api/workflow-builder/workflows/${workflowName}/v/${version}/approve`
        );
        return response.data;
    },

    // Activate workflow version
    activateWorkflow: async (workflowName: string, version: number): Promise<{ message: string }> => {
        const response = await apiClient.post(
            `/api/workflow-builder/workflows/${workflowName}/v/${version}/activate`
        );
        return response.data;
    },

    // Delete workflow version
    deleteWorkflow: async (workflowName: string, version: number): Promise<{ message: string }> => {
        const response = await apiClient.delete(
            `/api/workflow-builder/workflows/${workflowName}/v/${version}`
        );
        return response.data;
    },

    // Execute workflow
    executeWorkflow: async (data: WorkflowExecutionCreate): Promise<WorkflowExecution> => {
        const response = await apiClient.post('/api/workflow-builder/executions', data);
        return response.data;
    },

    // Get execution details
    getExecution: async (runId: string): Promise<WorkflowExecution> => {
        const response = await apiClient.get(`/api/workflow-builder/executions/${runId}`);
        return response.data;
    },

    // Pause execution
    pauseExecution: async (runId: string): Promise<{ message: string }> => {
        const response = await apiClient.post(`/api/workflow-builder/executions/${runId}/pause`);
        return response.data;
    },

    // Resume execution
    resumeExecution: async (runId: string): Promise<{ message: string }> => {
        const response = await apiClient.post(`/api/workflow-builder/executions/${runId}/resume`);
        return response.data;
    },
};
