import { motion } from 'framer-motion'
import {
  Activity,
  Thermometer,
  Gauge,
  Droplets,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  Zap
} from 'lucide-react'

import { GlassCard, CardContent, CardHeader, CardTitle } from '@/components/ui/glass-card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

import useSystemStore from '@/stores/system-store'
import { formatDuration } from '@/lib/utils'
import { SpectralView } from '@/components/SpectralView'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: "easeOut"
    },
  },
}

export function Dashboard() {
  const { status, activeProcesses, alerts } = useSystemStore()

  const recentAlerts = alerts.slice(0, 5)
  const runningProcesses = activeProcesses.filter(p => p.status === 'running')

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8 p-4 md:p-8 pt-6"
    >
      <div className="flex items-center justify-between space-y-2">
        <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-r from-cyan-400 to-teal-400 bg-clip-text text-transparent">
          System Overview
        </h2>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-cyan-400 border-cyan-400/50">
            v2.0.0
          </Badge>
        </div>
      </div>

      {/* Status Overview */}
      <motion.div variants={itemVariants}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <GlassCard>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">System Status</CardTitle>
              <Activity className="h-4 w-4 text-cyan-400" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2">
                <Badge
                  variant={status?.overall === 'healthy' ? 'success' : 'destructive'}
                  className="capitalize shadow-lg shadow-cyan-500/20"
                >
                  {status?.overall || 'Unknown'}
                </Badge>
                <span className="text-sm text-muted-foreground font-mono">
                  {status && formatDuration(status.uptime)}
                </span>
              </div>
            </CardContent>
          </GlassCard>

          <GlassCard>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Temperature</CardTitle>
              <Thermometer className="h-4 w-4 text-rose-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-100">
                {status?.components.daq.temperature?.toFixed(1) || '--'}°C
              </div>
              <Progress value={((status?.components.daq.temperature || 0) / 40) * 100} className="h-1 mt-2 bg-slate-800" />
            </CardContent>
          </GlassCard>

          <GlassCard>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Pressure</CardTitle>
              <Gauge className="h-4 w-4 text-amber-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-100">
                {status?.components.pumps.pressure?.toFixed(2) || '--'} bar
              </div>
              <Progress value={((status?.components.pumps.pressure || 0) / 3) * 100} className="h-1 mt-2 bg-slate-800" />
            </CardContent>
          </GlassCard>

          <GlassCard>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">Flow Rate</CardTitle>
              <Droplets className="h-4 w-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-slate-100">
                {status?.components.pumps.flowRate?.toFixed(1) || '--'} mL/min
              </div>
              <div className="text-xs text-muted-foreground mt-1">Target: 2.0 mL/min</div>
            </CardContent>
          </GlassCard>
        </div>
      </motion.div>

      {/* Spectral View */}
      <motion.div variants={itemVariants}>
        <GlassCard className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="h-5 w-5 text-yellow-400" />
              <span>Real-time Spectrum</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <SpectralView />
          </CardContent>
        </GlassCard>
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Processes */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <GlassCard className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-cyan-400" />
                <span>Active Processes</span>
                <Badge variant="secondary" className="bg-slate-800">{runningProcesses.length}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {runningProcesses.length > 0 ? (
                <div className="space-y-6">
                  {runningProcesses.map((process) => (
                    <div key={process.id} className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                          <span className="font-medium text-slate-200">{process.recipeName}</span>
                        </div>
                        <span className="text-sm text-muted-foreground font-mono">
                          Step {process.currentStep}/{process.totalSteps}
                        </span>
                      </div>
                      <Progress value={process.progress} className="h-2 bg-slate-800" />
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>Started: {process.startTime.toLocaleTimeString()}</span>
                        <span>{process.progress.toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <div className="h-16 w-16 rounded-full bg-slate-800/50 flex items-center justify-center mb-4">
                    <CheckCircle className="h-8 w-8 text-slate-600" />
                  </div>
                  <p>No active processes</p>
                </div>
              )}
            </CardContent>
          </GlassCard>
        </motion.div>

        {/* Recent Alerts */}
        <motion.div variants={itemVariants}>
          <GlassCard className="h-full">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-amber-400" />
                <span>Recent Alerts</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {recentAlerts.length > 0 ? (
                <div className="space-y-4">
                  {recentAlerts.map((alert) => (
                    <div key={alert.id} className="relative pl-4 border-l-2 border-slate-800">
                      <div className="flex items-start justify-between mb-1">
                        <Badge
                          variant={
                            alert.type === 'error' ? 'destructive' :
                              alert.type === 'warning' ? 'warning' :
                                alert.type === 'success' ? 'success' : 'secondary'
                          }
                          className="text-[10px] px-1.5 py-0 h-5"
                        >
                          {alert.type}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground font-mono">
                          {alert.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="text-sm text-slate-300 leading-tight">{alert.message}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <div className="h-16 w-16 rounded-full bg-slate-800/50 flex items-center justify-center mb-4">
                    <CheckCircle className="h-8 w-8 text-green-500/50" />
                  </div>
                  <p>All systems normal</p>
                </div>
              )}
            </CardContent>
          </GlassCard>
        </motion.div>
      </div>

      {/* System Components Status */}
      <motion.div variants={itemVariants}>
        <GlassCard>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-purple-400" />
              <span>Component Health</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {status?.components && Object.entries(status.components).map(([key, component]) => (
                <div key={key} className="bg-slate-900/40 rounded-lg p-3 text-center border border-slate-800/50 hover:border-cyan-500/30 transition-colors">
                  <div className="capitalize font-medium text-sm text-slate-300 mb-2">{key}</div>
                  <div className="flex items-center justify-center space-x-2">
                    <div className={`h-2 w-2 rounded-full ${component.status === 'online' ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' :
                      component.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                      }`} />
                    <span className="text-xs text-muted-foreground uppercase tracking-wider">{component.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </GlassCard>
      </motion.div>
    </motion.div>
  )
}