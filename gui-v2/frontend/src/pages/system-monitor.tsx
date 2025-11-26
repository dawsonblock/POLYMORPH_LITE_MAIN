import { motion } from 'framer-motion'
import {
  Activity,
  Database,
  HardDrive,
  Server,
  Cpu,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Bot,
  ZapOff
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

import useSystemStore from '@/stores/system-store'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

export function SystemMonitor() {
  const { status, isConnected } = useSystemStore()

  const getIcon = (name: string) => {
    if (name.toLowerCase().includes('database')) return <Database className="h-5 w-5" />
    if (name.toLowerCase().includes('file')) return <HardDrive className="h-5 w-5" />
    if (name.toLowerCase().includes('daq')) return <Activity className="h-5 w-5" />
    if (name.toLowerCase().includes('raman')) return <Server className="h-5 w-5" />
    if (name.toLowerCase().includes('ai')) return <Bot className="h-5 w-5" />
    return <Cpu className="h-5 w-5" />
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'healthy': return 'success'
      case 'warning': return 'warning'
      case 'error':
      case 'offline':
      case 'critical': return 'destructive'
      default: return 'secondary'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'error':
      case 'offline':
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  if (!status) {
    return (
      <div className="flex items-center justify-center h-[50vh]">
        <div className="text-center space-y-4">
          <Activity className="h-12 w-12 animate-pulse text-muted-foreground mx-auto" />
          <p className="text-muted-foreground">Waiting for system telemetry...</p>
        </div>
      </div>
    )
  }

  const components = [
    { name: 'DAQ Driver', ...status.components.daq },
    { name: 'Raman Driver', ...status.components.raman },
    { name: 'AI Service', ...status.components.ai },
    { name: 'Safety System', ...status.components.safety },
  ]

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">System Monitor</h2>
        {!isConnected && (
          <Badge variant="destructive" className="animate-pulse">
            Backend Disconnected
          </Badge>
        )}
      </div>

      {/* AI Circuit Breaker Warning */}
      {status.components.ai?.circuit_open && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mb-6"
        >
          <Alert variant="destructive" className="border-2 border-red-500 bg-red-50 dark:bg-red-900/20">
            <ZapOff className="h-5 w-5" />
            <AlertTitle className="text-lg font-bold">AI CIRCUIT BREAKER OPEN</AlertTitle>
            <AlertDescription className="text-base">
              The AI Service has failed {status.components.ai.failures} times consecutively.
              Automated safety protocols have triggered a failsafe.
              The system is operating in manual fallback mode.
            </AlertDescription>
          </Alert>
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {components.map((comp, idx) => (
          <motion.div key={idx} variants={itemVariants}>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium flex items-center space-x-2">
                  {getIcon(comp.name)}
                  <span>{comp.name}</span>
                </CardTitle>
                {getStatusIcon(comp.status)}
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between mt-2">
                  <Badge variant={getStatusColor(comp.status)} className="capitalize">
                    {comp.status}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Last update: {new Date(comp.lastUpdate || status.lastUpdate).toLocaleTimeString()}
                  </span>
                </div>

                <pre className="text-xs overflow-auto max-h-40 mt-4">
                  {JSON.stringify({
                    ...status,
                    lastUpdate: new Date().toISOString() // Add timestamp for display
                  }, null, 2)}
                </pre>

                <div className="mt-4 space-y-2">
                  <div className="text-xs font-medium text-muted-foreground">Details</div>
                  <ScrollArea className="h-[80px] w-full rounded-md border p-2">
                    <pre className="text-xs">
                      {JSON.stringify(comp, (key, value) => {
                        if (key === 'name' || key === 'status' || key === 'lastUpdate') return undefined
                        return value
                      }, 2)}
                    </pre>
                  </ScrollArea>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}