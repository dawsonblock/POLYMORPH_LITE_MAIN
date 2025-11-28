import { motion } from 'framer-motion'
import { GitBranch } from 'lucide-react'
import { Link } from 'react-router-dom'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useWorkflowCards } from '@/hooks/use-workflows'

export function WorkflowsPage() {
  const { data, isLoading, error } = useWorkflowCards()

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="h-32 animate-pulse bg-muted/40" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <Card className="border-destructive/40 bg-destructive/10">
        <CardHeader>
          <CardTitle className="text-destructive">Failed to load workflows</CardTitle>
          <CardDescription>{(error as Error).message}</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  if (!data || data.length === 0) {
    return (
      <Card className="border-dashed">
        <CardHeader>
          <CardTitle>No workflows yet</CardTitle>
          <CardDescription>
            Once workflows are configured and executed, they will appear here with key analytics.
          </CardDescription>
        </CardHeader>
      </Card>
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
          <h1 className="text-3xl font-bold tracking-tight">Workflows</h1>
          <p className="text-muted-foreground">
            Overview of workflow health, throughput, and reliability.
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {data.map(card => {
          const successPct = Math.round((card.success_rate ?? 0) * 100)
          const total = card.total_runs || 0
          const runningFrac = total ? card.running / total : 0
          const failedFrac = total ? card.failed / total : 0
          const completedFrac = total ? card.completed / total : 0

          return (
            <Link
              key={card.workflow_name}
              to={`/workflows/${encodeURIComponent(card.workflow_name)}`}
              className="block"
            >
              <Card className="relative overflow-hidden transition hover:ring-2 hover:ring-primary/40">
                <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
                  <div className="space-y-1">
                    <CardTitle className="flex items-center gap-2 text-sm font-medium">
                      <GitBranch className="h-4 w-4 text-muted-foreground" />
                      <span className="truncate">{card.workflow_name}</span>
                    </CardTitle>
                    <CardDescription>
                      {card.total_runs} total runs ·{' '}
                      {card.last_run_status ? `Last: ${card.last_run_status}` : 'No runs yet'}
                    </CardDescription>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {successPct}% success
                  </Badge>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Status mix</span>
                    <span>
                      {card.running} running · {card.completed} completed · {card.failed} failed
                    </span>
                  </div>
                  <div className="flex h-1.5 overflow-hidden rounded-full bg-muted/40">
                    <div className="bg-amber-400" style={{ width: `${runningFrac * 100}%` }} />
                    <div className="bg-emerald-500" style={{ width: `${completedFrac * 100}%` }} />
                    <div className="bg-rose-500" style={{ width: `${failedFrac * 100}%` }} />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                      Median:{' '}
                      {card.median_duration_seconds != null
                        ? `${card.median_duration_seconds.toFixed(1)}s`
                        : '—'}
                    </span>
                    <span>
                      P95:{' '}
                      {card.p95_duration_seconds != null
                        ? `${card.p95_duration_seconds.toFixed(1)}s`
                        : '—'}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </Link>
          )
        })}
      </div>
    </motion.div>
  )
}
