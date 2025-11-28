import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  useRecentExecutions,
  useWorkflowCard,
  useExecuteWorkflow,
  useWorkflowExecutionSummary,
} from '@/hooks/use-workflows'

export function WorkflowDetailsPage() {
  const params = useParams()
  const workflowName = params.workflowName as string | undefined

  const { data: card, isLoading, error } = useWorkflowCard(workflowName)
  const { data: recent } = useRecentExecutions({
    workflow_name: workflowName,
    limit: 20,
  })
  const { data: summary } = useWorkflowExecutionSummary(workflowName)
  const [paramsJson, setParamsJson] = useState('{}')
  const [metadataJson, setMetadataJson] = useState('{}')
  const execute = useExecuteWorkflow()

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading workflow...</p>
  }

  if (error || !card || !workflowName) {
    return (
      <Card className="border-destructive/40 bg-destructive/10">
        <CardHeader>
          <CardTitle className="text-destructive">Workflow not found</CardTitle>
          <CardDescription>{(error as Error)?.message}</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const successPct = Math.round((card.success_rate ?? 0) * 100)

  const handleStartRun = () => {
    let params: any = {}
    let metadata: any = {}
    try {
      params = paramsJson ? JSON.parse(paramsJson) : {}
    } catch {
      alert('Parameters JSON is invalid')
      return
    }
    try {
      metadata = metadataJson ? JSON.parse(metadataJson) : {}
    } catch {
      alert('Metadata JSON is invalid')
      return
    }

    execute.mutate(
      {
        workflow_name: workflowName!,
        parameters: params,
        metadata,
      },
      {
        onSuccess: newRun => {
          window.location.href = `/runs/${newRun.run_id}`
        },
      },
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{card.workflow_name}</h1>
          <p className="text-muted-foreground">
            Detailed view with recent runs and performance metrics.
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            <Link
              to={`/runs?workflow_name=${encodeURIComponent(card.workflow_name)}&status=failed`}
              className="text-primary underline-offset-2 hover:underline"
            >
              View failed runs in Runs
            </Link>
          </p>
        </div>
        <Badge variant="outline" className="text-xs">
          {successPct}% success
        </Badge>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Run summary</CardTitle>
            <CardDescription>Total executions and status mix.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Total runs</span>
              <span className="font-medium">{card.total_runs}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Running</span>
              <span className="font-medium">{card.running}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Completed</span>
              <span className="font-medium">{card.completed}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Failed</span>
              <span className="font-medium">{card.failed}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Durations</CardTitle>
            <CardDescription>Central tendency and tail latency.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Average</span>
              <span className="font-medium">
                {card.average_duration_seconds != null
                  ? `${card.average_duration_seconds.toFixed(1)}s`
                  : '—'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Median</span>
              <span className="font-medium">
                {card.median_duration_seconds != null
                  ? `${card.median_duration_seconds.toFixed(1)}s`
                  : '—'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">P95</span>
              <span className="font-medium">
                {card.p95_duration_seconds != null
                  ? `${card.p95_duration_seconds.toFixed(1)}s`
                  : '—'}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Last run</CardTitle>
            <CardDescription>Most recent execution status.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Status</span>
              <span className="font-medium">{card.last_run_status ?? '—'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Started</span>
              <span className="font-medium">{card.last_run_started_at ?? '—'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Completed</span>
              <span className="font-medium">{card.last_run_completed_at ?? '—'}</span>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Runs by operator</CardTitle>
            <CardDescription>Distribution of executions by operator.</CardDescription>
          </CardHeader>
          <CardContent>
            {summary && Object.keys(summary.runs_per_operator).length > 0 ? (
              <div className="space-y-2 text-xs">
                {Object.entries(summary.runs_per_operator)
                  .sort(([, a], [, b]) => (b as number) - (a as number))
                  .map(([operator, count]) => {
                    const max = Math.max(...Object.values(summary.runs_per_operator)) || 1
                    const width = `${(Number(count) / max) * 100}%`
                    return (
                      <div key={operator} className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="truncate text-muted-foreground">{operator}</span>
                          <span className="font-medium">{count}</span>
                        </div>
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/40">
                          <div className="h-full rounded-full bg-primary/80" style={{ width }} />
                        </div>
                      </div>
                    )
                  })}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No operator breakdown available yet.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Start new run</CardTitle>
          <CardDescription>
            Launch a new execution of this workflow with optional parameters and metadata.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Parameters (JSON)
              </label>
              <textarea
                className="h-28 w-full resize-none rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] text-foreground focus:outline-none"
                value={paramsJson}
                onChange={e => setParamsJson(e.target.value)}
                aria-label="Workflow parameters JSON"
                placeholder='{"input_volume_ml": 5, "temperature_c": 25}'
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] font-medium uppercase text-muted-foreground">
                Metadata (JSON)
              </label>
              <textarea
                className="h-28 w-full resize-none rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] text-foreground focus:outline-none"
                value={metadataJson}
                onChange={e => setMetadataJson(e.target.value)}
                aria-label="Run metadata JSON"
                placeholder='{"batch": "BATCH-001", "sample_id": "S1"}'
              />
            </div>
          </div>
          <div className="flex justify-end">
            <Button size="sm" onClick={handleStartRun} disabled={execute.isPending}>
              {execute.isPending ? 'Starting…' : 'Start run'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recent runs</CardTitle>
          <CardDescription>Most recent executions for this workflow.</CardDescription>
        </CardHeader>
        <CardContent>
          {recent && recent.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead className="bg-muted/50 text-[11px] uppercase text-muted-foreground">
                  <tr>
                    <th className="px-2 py-2 text-left">Run ID</th>
                    <th className="px-2 py-2 text-left">Status</th>
                    <th className="px-2 py-2 text-left">Operator</th>
                    <th className="px-2 py-2 text-left">Started</th>
                    <th className="px-2 py-2 text-left">Duration</th>
                    <th className="px-2 py-2 text-left">Metadata</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-muted/40">
                  {recent.map(row => (
                    <tr key={row.run_id} className="hover:bg-muted/40">
                      <td className="px-2 py-1.5">
                        <Link
                          to={`/runs/${row.run_id}`}
                          className="font-mono text-[11px] text-primary underline-offset-2 hover:underline"
                        >
                          {row.run_id.slice(0, 8)}…
                        </Link>
                      </td>
                      <td className="px-2 py-1.5">{row.status}</td>
                      <td className="px-2 py-1.5 text-muted-foreground">{row.operator ?? '—'}</td>
                      <td className="px-2 py-1.5 text-muted-foreground">{row.started_at ?? '—'}</td>
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
          ) : (
            <p className="text-sm text-muted-foreground">No recent runs for this workflow.</p>
          )}
        </CardContent>
      </Card>
    </motion.div>
  )
}
