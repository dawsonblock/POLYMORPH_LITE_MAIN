import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  Cpu,
  HardDrive,
  Clock,
  Play,
  AlertTriangle,
  CheckCircle2,
  Thermometer
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useRecentExecutions } from '@/hooks/use-workflows'

type StatusFilter = 'all' | 'running' | 'completed' | 'failed'

export function Dashboard() {
  const [systemStats, setSystemStats] = useState({
    cpu: 0,
    memory: 0,
    disk: 0,
    uptime: '0h 0m'
  })
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')

  // Mock data updates (replace with real websocket data later)
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemStats(prev => ({
        cpu: Math.floor(Math.random() * 30) + 10,
        memory: Math.floor(Math.random() * 20) + 40,
        disk: 45,
        uptime: '24h 12m'
      }))
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const item = {
    hidden: { y: 20, opacity: 0 },
    show: { y: 0, opacity: 1 }
  }

  const { data: recentRuns, isLoading: isLoadingRecent, error: recentError } = useRecentExecutions({
    limit: 5,
  })

  const rawRecentRuns = recentRuns ?? []
  const filteredRecentRuns =
    statusFilter === 'all'
      ? rawRecentRuns
      : rawRecentRuns.filter(run => run.status === statusFilter)

  const durationValues = filteredRecentRuns
    .map(run => run.total_duration_seconds)
    .filter((v): v is number => v != null)
    .sort((a, b) => a - b)

  const totalFiltered = filteredRecentRuns.length
  const completedFiltered = filteredRecentRuns.filter(run => run.status === 'completed').length

  let successRateDisplay: string | null = null
  let medianDisplay: string | null = null
  let p95Display: string | null = null

  if (totalFiltered > 0) {
    successRateDisplay = `${Math.round((completedFiltered / totalFiltered) * 100)}%`
  }

  if (durationValues.length > 0) {
    const mid = Math.floor(durationValues.length / 2)
    const median =
      durationValues.length % 2 === 1
        ? durationValues[mid]
        : (durationValues[mid - 1] + durationValues[mid]) / 2
    const p95Index = Math.floor(0.95 * (durationValues.length - 1))
    const p95 = durationValues[p95Index]
    medianDisplay = `${median.toFixed(1)}s`
    p95Display = `${p95.toFixed(1)}s`
  }

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">System overview and control center</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">View Logs</Button>
          <Button variant="scientific">
            <Play className="mr-2 h-4 w-4" />
            New Run
          </Button>
        </div>
      </div>

      {/* System Health Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.cpu}%</div>
              <p className="text-xs text-muted-foreground">
                <span className="text-emerald-500">Normal</span> load
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Memory</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.memory}%</div>
              <p className="text-xs text-muted-foreground">
                8GB / 16GB used
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Disk Space</CardTitle>
              <HardDrive className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.disk}%</div>
              <p className="text-xs text-muted-foreground">
                256GB free
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Uptime</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemStats.uptime}</div>
              <p className="text-xs text-muted-foreground">
                Since last reboot
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">

        {/* Active Run / Status */}
        <motion.div variants={item} className="col-span-4">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Active Workflow</CardTitle>
              <CardDescription>Current execution status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center h-[200px] border-2 border-dashed rounded-lg border-muted">
                <p className="text-muted-foreground">No active workflow</p>
                <Button variant="link" className="mt-2">Select a recipe to start</Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent Activity / Alerts */}
        <motion.div variants={item} className="col-span-3">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>System Alerts</CardTitle>
              <CardDescription>Recent notifications and warnings</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start gap-4 rounded-md border p-3 bg-background/50">
                  <CheckCircle2 className="mt-0.5 h-5 w-5 text-emerald-500" />
                  <div className="space-y-1">
                    <p className="text-sm font-medium leading-none">System Startup</p>
                    <p className="text-xs text-muted-foreground">All systems initialized successfully.</p>
                    <p className="text-xs text-muted-foreground mt-1">2 mins ago</p>
                  </div>
                </div>

                <div className="flex items-start gap-4 rounded-md border p-3 bg-background/50">
                  <Thermometer className="mt-0.5 h-5 w-5 text-sky-500" />
                  <div className="space-y-1">
                    <p className="text-sm font-medium leading-none">Temperature Stabilized</p>
                    <p className="text-xs text-muted-foreground">Reactor reached setpoint 25.0°C.</p>
                    <p className="text-xs text-muted-foreground mt-1">15 mins ago</p>
                  </div>
                </div>

                <div className="flex items-start gap-4 rounded-md border p-3 bg-amber-500/10 border-amber-500/20">
                  <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-500" />
                  <div className="space-y-1">
                    <p className="text-sm font-medium leading-none text-amber-500">Calibration Due</p>
                    <p className="text-xs text-muted-foreground">Raman spectrometer calibration expires in 2 days.</p>
                    <p className="text-xs text-muted-foreground mt-1">1 hour ago</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Recent workflow runs */}
        <motion.div variants={item} className="col-span-3">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Recent workflow runs</CardTitle>
              <CardDescription>Latest executions across all workflows.</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingRecent && (
                <p className="text-sm text-muted-foreground">Loading recent runs...</p>
              )}
              {recentError && (
                <p className="text-xs text-destructive">
                  {(recentError as Error).message}
                </p>
              )}
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex gap-1">
                  {(['all', 'running', 'completed', 'failed'] as StatusFilter[]).map(s => {
                    const isActive = statusFilter === s
                    return (
                      <Button
                        key={s}
                        type="button"
                        variant={isActive ? 'default' : 'outline'}
                        size="sm"
                        className="text-[11px]"
                        onClick={() => setStatusFilter(s)}
                      >
                        {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
                      </Button>
                    )
                  })}
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {successRateDisplay && <span>Success {successRateDisplay}</span>}
                  {medianDisplay && (
                    <span className={successRateDisplay ? 'ml-2' : ''}>Median {medianDisplay}</span>
                  )}
                  {p95Display && <span className="ml-2">P95 {p95Display}</span>}
                </div>
              </div>
              {filteredRecentRuns.length > 0 ? (
                <div className="space-y-2 text-xs">
                  {filteredRecentRuns.map(run => (
                    <div
                      key={run.run_id}
                      className="flex items-center justify-between rounded-md border border-border/60 bg-background/40 px-2 py-1.5"
                    >
                      <div className="space-y-0.5">
                        <a
                          href={`/runs/${run.run_id}`}
                          className="font-mono text-[11px] text-primary underline-offset-2 hover:underline"
                        >
                          {run.run_id.slice(0, 8)}…
                        </a>
                        <p className="text-[11px] text-muted-foreground">
                          {run.workflow_name} · {run.operator ?? 'Unknown'}
                        </p>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <DashboardStatusBadge status={run.status} />
                        <span className="text-[10px] text-muted-foreground">
                          {run.started_at ?? '—'}
                        </span>
                      </div>
                    </div>
                  ))}
                  <div className="pt-1 text-right text-[11px]">
                    <a
                      href={`/runs${
                        statusFilter === 'all' ? '' : `?status=${encodeURIComponent(statusFilter)}`
                      }`}
                      className="text-primary underline-offset-2 hover:underline"
                    >
                      View all in Runs
                    </a>
                  </div>
                </div>
              ) : !isLoadingRecent && !recentError ? (
                <p className="text-sm text-muted-foreground">No recent workflow runs.</p>
              ) : null}
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}

function DashboardStatusBadge({ status }: { status: string }) {
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