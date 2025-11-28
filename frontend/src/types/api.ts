// API Type Definitions
export interface LoginRequest {
    email: string;
    password: string;
}

export interface LoginResponse {
    access_token: string;
    token_type: string;
}

export interface User {
    email: string;
    role: string;
    permissions: string[];
}

export interface Sample {
    id: string;
    sample_id: string;
    status: string;
    created_at: string;
    extra_data?: Record<string, any>;
    parent_id?: string;
}

export interface InventoryItem {
    id: string;
    item_id: string;
    name: string;
    category: string;
    stock_quantity: number;
    min_stock_level: number;
    unit: string;
}

export interface CalibrationEntry {
    id: string;
    device_id: string;
    calibration_date: string;
    performed_by: string;
    status: string;
    notes?: string;
}

export interface WorkflowExecution {
    id: string;
    run_id: string;
    workflow_name: string;
    status: string;
    started_at: string;
    completed_at?: string;
    operator: string;
}

export interface ApiError {
    detail: string;
}
