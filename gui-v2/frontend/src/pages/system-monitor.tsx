import { useEffect, useState } from 'react'
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
  Bot
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'

interface ComponentHealth {
  name: string
  status: 'healthy' | 'warning' | 'error' | 'not_initialized' | 'critical'
  last_check: string
  response_time_ms: number
  error_message?: string
  details: Record<string, any>
}

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
  const [components, setComponents] = useState<ComponentHealth[]>([])
  const [error, setError] = useState<string | null>(null)

  const fetchHealth = async () => {
    try {
      const res = await fetch('http://localhost:8001/health/components')
      if (!res.ok) throw new Error('Failed to fetch health data')
      const data = await res.json()
      setComponents(data)
      setError(null)
    } catch (err) {
      console.error(err)
      setError('Could not load system health data. Is the backend running?')
      // Fallback mock data for demo purposes if backend is offline
      setComponents([
        {
          name: "DAQ Driver",
          status: "healthy",
          last_check: new Date().toISOString(),
          response_time_ms: 12.5,
          details: { last_voltage: 1.23 }
        },
        {
          name: "Raman Driver",
          status: "healthy",
          last_check: new Date().toISOString(),
          response_time_ms: 45.2,
          details: { last_peak_nm: 532.0 }
        },
        {
          name: "Database",
          status: "healthy",
          last_check: new Date().toISOString(),
          response_time_ms: 2.1,
          details: { path: "/data/system.db" }
        },
        {
          name: "File System",
          status: "healthy",
          last_check: new Date().toISOString(),
          response_time_ms: 0.5,
          details: { free_percent: 45.2 }
        },
        {
          name: "AI Service",
          status: "healthy",
          last_check: new Date().toISOString(),
          response_time_ms: 150.0,
          details: { url: "http://localhost:3000/infer" }
        }
      ])
    }
  }

  useEffect(() => {
    fetchHealth()
    const interval = setInterval(fetchHealth, 5000) // Poll every 5s
    return () => clearInterval(interval)
  }, [])

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
      case 'healthy': return 'success'
      case 'warning': return 'warning'
      case 'error':
      case 'critical': return 'destructive'
      default: return 'secondary'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />
      case 'error':
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">System Monitor</h2>
        {error && (
          <Badge variant="destructive" className="animate-pulse">
            Backend Offline - Showing Cached/Mock Data
          </Badge>
        )}
      </div>

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
                    {comp.status.replace('_', ' ')}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {comp.response_time_ms.toFixed(1)}ms
                  </span>
                </div>

                {comp.error_message && (
                  <div className="mt-3 text-sm text-red-500 bg-red-50 dark:bg-red-900/10 p-2 rounded">
                    {comp.error_message}
                  </div>
                )}

                <div className="mt-4 space-y-2">
                  <div className="text-xs font-medium text-muted-foreground">Details</div>
                  <ScrollArea className="h-[80px] w-full rounded-md border p-2">
                    <pre className="text-xs">
                      {JSON.stringify(comp.details, null, 2)}
                    </pre>
                  </ScrollArea>
                </div>

                <div className="mt-2 text-xs text-muted-foreground text-right">
                  Last check: {new Date(comp.last_check).toLocaleTimeString()}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}