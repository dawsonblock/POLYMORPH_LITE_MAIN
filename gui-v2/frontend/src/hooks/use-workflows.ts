import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

import type {
  ExecutionsFilter,
  RecentExecutionsParams,
  WorkflowExecutionCreate,
  WorkflowRerunRequest,
} from '@/api/workflow-builder'
import {
  getWorkflowCards,
  getWorkflowCard,
  getRecentExecutions,
  getExecutions,
  getExecution,
  getWorkflowExecutionSummary,
  getComplianceRun,
  executeWorkflow,
  rerunExecution,
  pauseExecution,
  resumeExecution,
  abortExecution,
} from '@/api/workflow-builder'

export function useWorkflowCards() {
  return useQuery({
    queryKey: ['workflowCards'],
    queryFn: getWorkflowCards,
  })
}

export function useWorkflowCard(workflowName?: string) {
  return useQuery({
    queryKey: ['workflowCard', workflowName],
    queryFn: () => getWorkflowCard(workflowName!),
    enabled: !!workflowName,
  })
}

export function useRecentExecutions(params: RecentExecutionsParams) {
  return useQuery({
    queryKey: ['recentExecutions', params],
    queryFn: () => getRecentExecutions(params),
  })
}

export function useExecutions(filter: ExecutionsFilter) {
  return useQuery({
    queryKey: ['executions', filter],
    queryFn: () => getExecutions(filter),
  })
}

export function useExecution(runId?: string) {
  return useQuery({
    queryKey: ['execution', runId],
    queryFn: () => getExecution(runId!),
    enabled: !!runId,
  })
}

export function useWorkflowExecutionSummary(workflowName?: string) {
  return useQuery({
    queryKey: ['workflowSummary', workflowName],
    queryFn: () => getWorkflowExecutionSummary(workflowName!),
    enabled: !!workflowName,
  })
}

export function useComplianceRun(runId?: string) {
  return useQuery({
    queryKey: ['complianceRun', runId],
    queryFn: () => getComplianceRun(runId!),
    enabled: !!runId,
  })
}

export function useExecuteWorkflow() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: WorkflowExecutionCreate) => executeWorkflow(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['executions'] })
      qc.invalidateQueries({ queryKey: ['workflowCards'] })
    },
  })
}

export function useRerunExecution(runId?: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: WorkflowRerunRequest) => rerunExecution(runId!, body),
    onSuccess: (newRun) => {
      qc.invalidateQueries({ queryKey: ['executions'] })
      qc.invalidateQueries({ queryKey: ['recentExecutions'] })
      qc.invalidateQueries({ queryKey: ['workflowCards'] })
      qc.setQueryData(['execution', newRun.run_id], newRun)
    },
  })
}

export function usePauseExecution(runId?: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => pauseExecution(runId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['execution', runId] })
      qc.invalidateQueries({ queryKey: ['executions'] })
      qc.invalidateQueries({ queryKey: ['recentExecutions'] })
    },
  })
}

export function useResumeExecution(runId?: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => resumeExecution(runId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['execution', runId] })
      qc.invalidateQueries({ queryKey: ['executions'] })
      qc.invalidateQueries({ queryKey: ['recentExecutions'] })
    },
  })
}

export function useAbortExecution(runId?: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => abortExecution(runId!),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['execution', runId] })
      qc.invalidateQueries({ queryKey: ['executions'] })
      qc.invalidateQueries({ queryKey: ['recentExecutions'] })
    },
  })
}
