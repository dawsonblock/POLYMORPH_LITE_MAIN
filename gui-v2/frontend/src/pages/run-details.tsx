import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { useAbortExecution, useComplianceRun, useExecution, usePauseExecution, useResumeExecution, useRerunExecution } from '@/hooks/use-workflows'

export function RunDetailsPage() {
  const params = useParams()
  const runId = params.runId as string | undefined

  const { data: exec, isLoading, error } = useExecution(runId)
  const { data: compliance } = useComplianceRun(runId)

  const pause = usePauseExecution(runId)
  const resume = useResumeExecution(runId)
  const abort = useAbortExecution(runId)
  const rerun = useRerunExecution(runId)

  const [metadataOverride, setMetadataOverride] = useState('{}')

  const disabled =
    pause.isPending ||
    resume.isPending ||
    abort.isPending ||
    rerun.isPending ||
    isLoading

  const onRerun = () => {
    let parsed: any = {}
    try {
      parsed = JSON.parse(metadataOverride || '{}')
    } catch {
      alert('Metadata override must be valid JSON')
      return
    }
    rerun.mutate(
      { metadata_override: parsed },
      {
        onSuccess: newRun => {
          window.location.href = `/runs/${newRun.run_id}`
        },
      },
    )
  }

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading run...</p>
  }
  if (error || !exec) {
    return (
      <Card className="border-destructive/40 bg-destructive/10">
        <CardHeader>
          <CardTitle className="text-destructive">Run not found</CardTitle>
          <CardDescription>{(error as Error)?.message}</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const duration =
    exec.total_duration_seconds != null
      ? `${exec.total_duration_seconds.toFixed(1)}s`
      : '—'

  let statusVariant: 'default' | 'success' | 'error' | 'outline' = 'outline'
  if (exec.status === 'completed') statusVariant = 'success'
  else if (exec.status === 'failed') statusVariant = 'error'
  else if (exec.status === 'running') statusVariant = 'default'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col gap-4 lg:flex-row lg:items-start"
    >
      <div className="flex-1 space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
            <div>
              <CardTitle className="text-base">
                Run {exec.run_id.slice(0, 8)}…
              </CardTitle>
              <CardDescription>
                {exec.workflow_name} · version {exec.workflow_version ?? 'latest'}
              </CardDescription>
            </div>
            <Badge variant={statusVariant}>{exec.status}</Badge>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-3 text-xs text-muted-foreground md:grid-cols-4">
              <div>
                <div className="text-[11px] uppercase">Operator</div>
                <div className="text-foreground">{exec.operator ?? 'Unknown'}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase">Started</div>
                <div>{exec.started_at ?? '—'}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase">Completed</div>
                <div>{exec.completed_at ?? '—'}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase">Duration</div>
                <div>{duration}</div>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => pause.mutate()} disabled={disabled}>
                Pause
              </Button>
              <Button variant="outline" size="sm" onClick={() => resume.mutate()} disabled={disabled}>
                Resume
              </Button>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => abort.mutate()}
                disabled={disabled}
              >
                Abort
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Run metadata</CardTitle>
            <CardDescription>Labels and contextual fields for this run.</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="max-h-64 overflow-auto rounded-md bg-black/80 p-3 text-[11px] text-muted-foreground">
              {JSON.stringify(exec.run_metadata, null, 2)}
            </pre>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Results / payload</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-64 overflow-auto rounded-md bg-black/80 p-3 text-[11px] text-muted-foreground">
              {JSON.stringify(exec.results ?? {}, null, 2)}
            </pre>
          </CardContent>
        </Card>
      </div>

      <div className="w-full max-w-sm space-y-4 lg:sticky lg:top-0">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Rerun with overrides</CardTitle>
            <CardDescription>
              Provide JSON overrides for run metadata (e.g. new batch or sample IDs). Leave empty
              to reuse original values.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <textarea
              className="mt-1 h-32 w-full resize-none rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] text-foreground focus:outline-none"
              value={metadataOverride}
              onChange={e => setMetadataOverride(e.target.value)}
              aria-label="Metadata override JSON"
              placeholder='{"batch": "NEW-BATCH"}'
            />
            <Button
              className="mt-3 w-full"
              size="sm"
              onClick={onRerun}
              disabled={disabled}
            >
              {rerun.isPending ? 'Starting rerun…' : 'Start rerun'}
            </Button>
          </CardContent>
        </Card>

        {compliance && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Compliance / audit trail</CardTitle>
              <CardDescription>
                Snapshot, approvals, and audit events for compliance review.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="max-h-64 overflow-auto rounded-md bg-black/80 p-3 text-[11px] text-muted-foreground">
                {JSON.stringify(compliance, null, 2)}
              </pre>
            </CardContent>
          </Card>
        )}
      </div>
    </motion.div>
  )
}
