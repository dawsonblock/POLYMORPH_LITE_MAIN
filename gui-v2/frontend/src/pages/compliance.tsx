import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  ShieldCheck,
  FileCheck,
  Users,
  Lock,
  Download,
  Filter
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

// Mock audit logs
const auditLogs = [
  { id: 1, timestamp: '2023-11-27 10:30:00', user: 'admin@polymorph.lab', action: 'LOGIN_SUCCESS', details: 'User logged in successfully', status: 'success' },
  { id: 2, timestamp: '2023-11-27 10:35:12', user: 'admin@polymorph.lab', action: 'RECIPE_CREATED', details: 'Created recipe "New Protocol"', status: 'success' },
  { id: 3, timestamp: '2023-11-27 11:00:00', user: 'system', action: 'SYSTEM_CHECK', details: 'Daily system health check passed', status: 'success' },
  { id: 4, timestamp: '2023-11-27 11:15:22', user: 'technician@polymorph.lab', action: 'LOGIN_FAILED', details: 'Invalid password attempt', status: 'error' },
  { id: 5, timestamp: '2023-11-27 11:15:45', user: 'technician@polymorph.lab', action: 'LOGIN_SUCCESS', details: 'User logged in successfully', status: 'success' },
]

export function Compliance() {
  const [searchTerm, setSearchTerm] = useState('')

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Compliance & Audit</h1>
          <p className="text-muted-foreground">21 CFR Part 11 Audit Trails and Security Logs</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <Download className="mr-2 h-4 w-4" />
            Export Report
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Audit Integrity</CardTitle>
            <ShieldCheck className="h-4 w-4 text-emerald-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Verified</div>
            <p className="text-xs text-muted-foreground">Last check: 5 mins ago</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Users</CardTitle>
            <Users className="h-4 w-4 text-sky-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">2 Admins, 1 Technician</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Security Events</CardTitle>
            <Lock className="h-4 w-4 text-amber-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">1</div>
            <p className="text-xs text-muted-foreground">Failed login attempt (24h)</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Audit Log</CardTitle>
              <CardDescription>Immutable record of all system actions</CardDescription>
            </div>
            <div className="flex gap-2">
              <div className="relative w-64">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search logs..."
                  className="pl-9"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              <Button variant="outline" size="icon">
                <Filter className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <table className="w-full text-sm text-left">
              <thead className="bg-muted/50 text-muted-foreground">
                <tr>
                  <th className="p-4 font-medium">Timestamp</th>
                  <th className="p-4 font-medium">User</th>
                  <th className="p-4 font-medium">Action</th>
                  <th className="p-4 font-medium">Details</th>
                  <th className="p-4 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {auditLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-muted/50 transition-colors">
                    <td className="p-4 font-mono text-xs">{log.timestamp}</td>
                    <td className="p-4">{log.user}</td>
                    <td className="p-4 font-medium">{log.action}</td>
                    <td className="p-4 text-muted-foreground">{log.details}</td>
                    <td className="p-4">
                      <Badge variant={log.status === 'success' ? 'success' : 'error'}>
                        {log.status === 'success' ? 'Success' : 'Failed'}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}