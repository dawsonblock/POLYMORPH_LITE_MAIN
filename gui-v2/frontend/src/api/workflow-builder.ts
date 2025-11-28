import useAuthStore from '@/stores/auth-store'

const API_ROOT = (import.meta as any).env?.VITE_API_BASE_URL ?? 'http://localhost:8000'
const WORKFLOW_BASE = `${API_ROOT}/api/workflow-builder`
const COMPLIANCE_BASE = `${API_ROOT}/api/compliance`

export type JsonMap = Record<string, any>

export interface WorkflowExecutionCreate {
  workflow_name: string
  workflow_version?: number | null
  parameters?: JsonMap
  metadata?: JsonMap
}

export interface WorkflowRerunRequest {
  parameters_override?: JsonMap | null
  metadata_override?: JsonMap | null
}

export interface WorkflowExecutionResponse {
  run_id: string
  workflow_name: string
  workflow_version: number | null
  status: string
  operator: string | null
  started_at: string | null
  completed_at: string | null
  total_duration_seconds?: number | null
  orchestrator_run_id?: string | null
  results?: JsonMap | null
  run_metadata: JsonMap
}

export interface WorkflowExecutionTableRow {
  run_id: string
  workflow_name: string
  workflow_version: number | null
  status: string
  operator: string | null
  started_at: string | null
  completed_at: string | null
  total_duration_seconds: number | null
  orchestrator_run_id?: string | null
  run_metadata: JsonMap
}

export interface WorkflowSummaryCard {
  workflow_name: string
  total_runs: number
  running: number
  completed: number
  failed: number
  average_duration_seconds: number | null
  median_duration_seconds: number | null
  p95_duration_seconds: number | null
  success_rate: number | null
  last_run_status: string | null
  last_run_started_at: string | null
  last_run_completed_at: string | null
}

export interface WorkflowExecutionSummaryResponse {
  workflow_name: string
  total: number
  by_status: Record<string, number>
  average_duration_seconds: number | null
  median_duration_seconds: number | null
  p95_duration_seconds: number | null
  success_rate: number | null
  runs_per_operator: Record<string, number>
  last_run_status: string | null
  last_run_started_at: string | null
  last_run_completed_at: string | null
}

export interface ComplianceRunDetails {
  run_id: string
  execution: WorkflowExecutionResponse
  config_snapshot: JsonMap
  audits: JsonMap[]
  [key: string]: any
}

function authHeaders(): Record<string, string> {
  const token = useAuthStore.getState().token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function http<T>(url: string, init?: RequestInit): Promise<T> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...authHeaders(),
    ...(init?.headers as HeadersInit | undefined),
  }

  const res = await fetch(url, {
    ...init,
    headers,
  })

  if (!res.ok) {
    const text = await res.text()
    try {
      const json = JSON.parse(text)
      throw new Error(json.detail || text || `HTTP ${res.status}`)
    } catch {
      throw new Error(text || `HTTP ${res.status}`)
    }
  }

  if (res.status === 204) {
    // no content
    return undefined as unknown as T
  }

  return (await res.json()) as T
}

// ---- Reads ----

export function getWorkflowCards() {
  return http<WorkflowSummaryCard[]>(`${WORKFLOW_BASE}/ui/workflows/cards`)
}

export function getWorkflowCard(workflowName: string) {
  return http<WorkflowSummaryCard>(
    `${WORKFLOW_BASE}/ui/workflows/${encodeURIComponent(workflowName)}/card`,
  )
}

export interface RecentExecutionsParams {
  workflow_name?: string
  limit?: number
}

export function getRecentExecutions(params: RecentExecutionsParams = {}) {
  const search = new URLSearchParams()
  if (params.workflow_name) search.set('workflow_name', params.workflow_name)
  if (params.limit != null) search.set('limit', String(params.limit))
  const qs = search.toString()
  return http<WorkflowExecutionTableRow[]>(
    `${WORKFLOW_BASE}/ui/recent-executions${qs ? `?${qs}` : ''}`,
  )
}

export interface ExecutionsFilter {
  workflow_name?: string
  status?: string
  operator?: string
  started_after?: string
  started_before?: string
  metadata_key?: string
  metadata_value?: string
  limit?: number
  offset?: number
}

export function getExecutions(filter: ExecutionsFilter = {}) {
  const search = new URLSearchParams()
  Object.entries(filter).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') {
      search.set(k, String(v))
    }
  })
  const qs = search.toString()
  return http<WorkflowExecutionResponse[]>(
    `${WORKFLOW_BASE}/executions${qs ? `?${qs}` : ''}`,
  )
}

export function getExecution(runId: string) {
  return http<WorkflowExecutionResponse>(
    `${WORKFLOW_BASE}/executions/${encodeURIComponent(runId)}`,
  )
}

export function getWorkflowExecutionSummary(workflowName: string) {
  return http<WorkflowExecutionSummaryResponse>(
    `${WORKFLOW_BASE}/workflows/${encodeURIComponent(
      workflowName,
    )}/executions/summary`,
  )
}

export function getComplianceRun(runId: string) {
  return http<ComplianceRunDetails>(
    `${COMPLIANCE_BASE}/run/${encodeURIComponent(runId)}`,
  )
}

// ---- Writes ----

export function executeWorkflow(body: WorkflowExecutionCreate) {
  return http<WorkflowExecutionResponse>(`${WORKFLOW_BASE}/execute`, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function rerunExecution(runId: string, body: WorkflowRerunRequest) {
  return http<WorkflowExecutionResponse>(
    `${WORKFLOW_BASE}/executions/${encodeURIComponent(runId)}/rerun`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  )
}

export function pauseExecution(runId: string) {
  return http<WorkflowExecutionResponse>(
    `${WORKFLOW_BASE}/executions/${encodeURIComponent(runId)}/pause`,
    { method: 'POST' },
  )
}

export function resumeExecution(runId: string) {
  return http<WorkflowExecutionResponse>(
    `${WORKFLOW_BASE}/executions/${encodeURIComponent(runId)}/resume`,
    { method: 'POST' },
  )
}

export function abortExecution(runId: string) {
  return http<WorkflowExecutionResponse>(
    `${WORKFLOW_BASE}/executions/${encodeURIComponent(runId)}/abort`,
    { method: 'POST' },
  )
}
