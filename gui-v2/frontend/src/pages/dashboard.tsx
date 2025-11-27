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
import { useWebSocket } from '@/hooks/use-websocket'

export function Dashboard() {
  const { lastMessage } = useWebSocket()
  const [systemStats, setSystemStats] = useState({
    cpu: 0,
    memory: 0,
    disk: 0,
    uptime: '0h 0m'
  })

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
      </div>
    </motion.div>
  )
}