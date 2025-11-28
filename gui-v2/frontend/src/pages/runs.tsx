import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { useLocation } from 'react-router-dom'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { useExecutions } from '@/hooks/use-workflows'
import type { ExecutionsFilter } from '@/api/workflow-builder'

type StatusFilter = 'all' | 'running' | 'completed' | 'failed'

export function RunsPage() {
  const location = useLocation()
  const searchParams = new URLSearchParams(location.search)

  const [workflowName, setWorkflowName] = useState(() => searchParams.get('workflow_name') ?? '')
  const [operator, setOperator] = useState(() => searchParams.get('operator') ?? '')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>(() => {
    const raw = searchParams.get('status') as StatusFilter | null
    return raw && ['all', 'running', 'completed', 'failed'].includes(raw) ? raw : 'all'
  })
  const [metadataKey, setMetadataKey] = useState(() => searchParams.get('metadata_key') ?? '')
  const [metadataValue, setMetadataValue] = useState(() => searchParams.get('metadata_value') ?? '')
  const [startedAfter, setStartedAfter] = useState(() => searchParams.get('started_after') ?? '')
  const [startedBefore, setStartedBefore] = useState(() => searchParams.get('started_before') ?? '')

  const effectiveFilter: ExecutionsFilter = useMemo(
    () => ({
      workflow_name: workflowName || undefined,
      operator: operator || undefined,
      status: statusFilter === 'all' ? undefined : statusFilter,
      metadata_key: metadataKey || undefined,
      metadata_value: metadataValue || undefined,
      started_after: startedAfter || undefined,
      started_before: startedBefore || undefined,
      limit: 100,
    }),
    [workflowName, operator, statusFilter, metadataKey, metadataValue, startedAfter, startedBefore],
  )

  const { data, isLoading, error } = useExecutions(effectiveFilter)

  const clearAll = () => {
    setWorkflowName('')
    setOperator('')
    setStatusFilter('all')
    setMetadataKey('')
    setMetadataValue('')
    setStartedAfter('')
    setStartedBefore('')
  }

  const hasActiveFilters =
    workflowName ||
    operator ||
    statusFilter !== 'all' ||
    metadataKey ||
    metadataValue ||
    startedAfter ||
    startedBefore

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Runs</h1>
          <p className="text-muted-foreground">
            Browse workflow executions by status, operator, time window, and metadata.
          </p>
        </div>
        {hasActiveFilters && (
          <Button variant="ghost" size="sm" onClick={clearAll}>
            Clear filters
          </Button>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Filters</CardTitle>
          <CardDescription>Refine which executions appear in the table below.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Workflow
              </label>
              <Input
                placeholder="workflow_name"
                value={workflowName}
                onChange={e => setWorkflowName(e.target.value)}
              />
            </div>

            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Operator
              </label>
              <Input
                placeholder="user@example.com"
                value={operator}
                onChange={e => setOperator(e.target.value)}
              />
            </div>

            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Status
              </label>
              <div className="flex gap-1">
                {(['all', 'running', 'completed', 'failed'] as StatusFilter[]).map(s => {
                  const isActive = statusFilter === s
                  return (
                    <Button
                      key={s}
                      type="button"
                      variant={isActive ? 'default' : 'outline'}
                      size="sm"
                      className="flex-1"
                      onClick={() => setStatusFilter(s)}
                    >
                      {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
                    </Button>
                  )
                })}
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Started after
              </label>
              <Input
                type="datetime-local"
                value={startedAfter}
                onChange={e => setStartedAfter(e.target.value)}
              />
            </div>

            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Started before
              </label>
              <Input
                type="datetime-local"
                value={startedBefore}
                onChange={e => setStartedBefore(e.target.value)}
              />
            </div>

            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Metadata filter
              </label>
              <div className="flex gap-1">
                <Input
                  placeholder="key (e.g. batch)"
                  value={metadataKey}
                  onChange={e => setMetadataKey(e.target.value)}
                />
                <Input
                  placeholder="value (e.g. B123)"
                  value={metadataValue}
                  onChange={e => setMetadataValue(e.target.value)}
                />
              </div>
              <p className="text-[10px] text-muted-foreground">
                Uses metadata_key / metadata_value filters in the API.
              </p>
            </div>
          </div>

          {hasActiveFilters && (
            <div className="flex flex-wrap gap-2 text-[11px]">
              {workflowName && (
                <FilterChip label={`Workflow: ${workflowName}`} onClear={() => setWorkflowName('')} />
              )}
              {operator && (
                <FilterChip label={`Operator: ${operator}`} onClear={() => setOperator('')} />
              )}
              {statusFilter !== 'all' && (
                <FilterChip label={`Status: ${statusFilter}`} onClear={() => setStatusFilter('all')} />
              )}
              {startedAfter && (
                <FilterChip
                  label={`Started ≥ ${startedAfter}`}
                  onClear={() => setStartedAfter('')}
                />
              )}
              {startedBefore && (
                <FilterChip
                  label={`Started ≤ ${startedBefore}`}
                  onClear={() => setStartedBefore('')}
                />
              )}
              {metadataKey && (
                <FilterChip
                  label={`Metadata ${metadataKey}=${metadataValue || '∗'}`}
                  onClear={() => {
                    setMetadataKey('')
                    setMetadataValue('')
                  }}
                />
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Executions</CardTitle>
          <CardDescription>Recent workflow runs matching the current filters.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading && <p className="text-sm text-muted-foreground">Loading runs...</p>}
          {error && (
            <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
              {(error as Error).message}
            </div>
          )}

          {data && data.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="bg-muted/50 text-[11px] uppercase text-muted-foreground">
                  <tr>
                    <th className="px-2 py-2 text-left">Run ID</th>
                    <th className="px-2 py-2 text-left">Workflow</th>
                    <th className="px-2 py-2 text-left">Status</th>
                    <th className="px-2 py-2 text-left">Operator</th>
                    <th className="px-2 py-2 text-left">Started</th>
                    <th className="px-2 py-2 text-left">Duration</th>
                    <th className="px-2 py-2 text-left">Metadata</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-muted/40">
                  {data.map(row => (
                    <tr key={row.run_id} className="hover:bg-muted/40">
                      <td className="px-2 py-1.5">
                        <a
                          href={`/runs/${row.run_id}`}
                          className="font-mono text-[11px] text-primary underline-offset-2 hover:underline"
                        >
                          {row.run_id.slice(0, 8)}…
                        </a>
                      </td>
                      <td className="px-2 py-1.5 text-foreground">{row.workflow_name}</td>
                      <td className="px-2 py-1.5">
                        <StatusBadge status={row.status} />
                      </td>
                      <td className="px-2 py-1.5 text-muted-foreground">
                        {row.operator ?? '—'}
                      </td>
                      <td className="px-2 py-1.5 text-muted-foreground">
                        {row.started_at ?? '—'}
                      </td>
                      <td className="px-2 py-1.5 text-muted-foreground">
                        {row.total_duration_seconds != null
                          ? `${row.total_duration_seconds.toFixed(1)}s`
                          : '—'}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-[11px] text-muted-foreground">
                        {row.run_metadata && Object.keys(row.run_metadata).length
                          ? JSON.stringify(row.run_metadata)
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : !isLoading && !error ? (
            <p className="text-sm text-muted-foreground">
              No runs match the current filters.
            </p>
          ) : null}
        </CardContent>
      </Card>
    </motion.div>
  )
}

function FilterChip({ label, onClear }: { label: string; onClear: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground">
      {label}
      <button
        type="button"
        onClick={onClear}
        className="text-[9px] hover:text-foreground"
        aria-label="Remove filter"
      >
        ×
      </button>
    </span>
  )
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'completed') {
    return <Badge variant="success">Completed</Badge>
  }
  if (status === 'failed') {
    return <Badge variant="error">Failed</Badge>
  }
  if (status === 'running') {
    return <Badge variant="default">Running</Badge>
  }
  return <Badge variant="outline">{status}</Badge>
}
