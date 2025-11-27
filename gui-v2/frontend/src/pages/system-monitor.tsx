import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { RefreshCw, Thermometer, Activity, Zap } from 'lucide-react'

// Mock data generator
const generateData = (count: number) => {
  return Array.from({ length: count }).map((_, i) => ({
    time: i,
    temp: 20 + Math.random() * 5,
    pressure: 1 + Math.random() * 0.2,
    power: 50 + Math.random() * 10
  }))
}

export function SystemMonitor() {
  const [data, setData] = useState(generateData(20))
  const [isLive, setIsLive] = useState(true)

  useEffect(() => {
    if (!isLive) return
    const interval = setInterval(() => {
      setData(prev => {
        const next = [...prev.slice(1), {
          time: prev[prev.length - 1].time + 1,
          temp: 20 + Math.random() * 5,
          pressure: 1 + Math.random() * 0.2,
          power: 50 + Math.random() * 10
        }]
        return next
      })
    }, 1000)
    return () => clearInterval(interval)
  }, [isLive])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Monitor</h1>
          <p className="text-muted-foreground">Real-time hardware telemetry</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={isLive ? "destructive" : "default"}
            onClick={() => setIsLive(!isLive)}
          >
            {isLive ? "Pause Stream" : "Resume Stream"}
          </Button>
          <Button variant="outline" size="icon">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Temperature</CardTitle>
            <Thermometer className="h-4 w-4 text-rose-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data[data.length - 1].temp.toFixed(1)}°C</div>
            <Badge variant="success" pulsing className="mt-1">Stable</Badge>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pressure</CardTitle>
            <Activity className="h-4 w-4 text-sky-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data[data.length - 1].pressure.toFixed(2)} bar</div>
            <Badge variant="success" pulsing className="mt-1">Nominal</Badge>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Power Draw</CardTitle>
            <Zap className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data[data.length - 1].power.toFixed(1)} W</div>
            <p className="text-xs text-muted-foreground mt-1">Peak: 62.4 W</p>
          </CardContent>
        </Card>
      </div>

      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Telemetry Stream</CardTitle>
          <CardDescription>Live sensor data aggregation</CardDescription>
        </CardHeader>
        <CardContent className="pl-2">
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                <XAxis
                  dataKey="time"
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value) => `${value}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    borderColor: 'hsl(var(--border))',
                    borderRadius: 'var(--radius)'
                  }}
                  itemStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line
                  type="monotone"
                  dataKey="temp"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="pressure"
                  stroke="#0ea5e9"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="power"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}