import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Settings as SettingsIcon,
  Save,
  Server,
  Users,
  Info,
  Plus,
} from 'lucide-react'
import { toast } from 'sonner'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'

import useAuthStore from '@/stores/auth-store'

interface User {
  email: string
  name: string
  role: string
  created: number
}

export function Settings() {
  const [activeTab, setActiveTab] = useState('general')
  const [users, setUsers] = useState<User[]>([])
  const [config, setConfig] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const { token, user } = useAuthStore()

  const fetchConfig = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/settings/config', {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (res.ok) setConfig(await res.json())
    } catch (e) {
      console.error(e)
    }
  }

  const fetchUsers = async () => {
    if (user?.role !== 'admin') return
    try {
      const res = await fetch('http://localhost:8000/api/settings/users', {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (res.ok) setUsers(await res.json())
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchConfig()
    if (activeTab === 'users') fetchUsers()
  }, [token, activeTab])

  const handleSaveConfig = async () => {
    setIsLoading(true)
    try {
      // Mock save for now
      await new Promise(resolve => setTimeout(resolve, 1000))
      toast.success('Configuration saved')
    } catch (e) {
      toast.error('Failed to save configuration')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
          <p className="text-muted-foreground">Manage system configuration and preferences.</p>
        </div>
        <Button onClick={handleSaveConfig} disabled={isLoading}>
          <Save className="mr-2 h-4 w-4" />
          {isLoading ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList>
          <TabsTrigger value="general" className="flex items-center space-x-2">
            <SettingsIcon className="h-4 w-4" />
            <span>General</span>
          </TabsTrigger>
          <TabsTrigger value="hardware" className="flex items-center space-x-2">
            <Server className="h-4 w-4" />
            <span>Hardware</span>
          </TabsTrigger>
          <TabsTrigger value="users" className="flex items-center space-x-2">
            <Users className="h-4 w-4" />
            <span>Users</span>
          </TabsTrigger>
          <TabsTrigger value="about" className="flex items-center space-x-2">
            <Info className="h-4 w-4" />
            <span>About</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="general">
          <Card>
            <CardHeader>
              <CardTitle>System Information</CardTitle>
              <CardDescription>
                General system settings and environment configuration.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>System Name</Label>
                  <Input defaultValue={config?.system?.name || 'POLYMORPH-4 Lite'} />
                </div>
                <div className="space-y-2">
                  <Label>Environment</Label>
                  <div className="flex items-center space-x-2 h-10">
                    <Badge variant="outline" className="text-cyan-500 border-cyan-500/50">
                      {config?.system?.environment || 'DEVELOPMENT'}
                    </Badge>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Data Directory</Label>
                  <Input defaultValue={config?.system?.data_dir || 'data'} readOnly className="bg-muted" />
                </div>
                <div className="space-y-2">
                  <Label>Log Level</Label>
                  <Input defaultValue={config?.system?.log_level || 'INFO'} />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="hardware">
          <Card>
            <CardHeader>
              <CardTitle>Hardware Configuration</CardTitle>
              <CardDescription>
                Driver settings and connection parameters.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium">DAQ System</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Backend Driver</Label>
                    <Input defaultValue={config?.daq?.backend || 'simulator'} />
                  </div>
                  <div className="space-y-2">
                    <Label>NI Device Name</Label>
                    <Input defaultValue={config?.daq?.ni_device_name || 'Dev1'} />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="text-lg font-medium">Raman Spectrometer</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Provider</Label>
                    <Input defaultValue={config?.raman?.provider || 'simulator'} />
                  </div>
                  <div className="space-y-2">
                    <Label>Integration Time (ms)</Label>
                    <Input type="number" defaultValue="100" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="users">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>User Management</CardTitle>
                  <CardDescription>
                    Manage system access and roles.
                  </CardDescription>
                </div>
                <Button size="sm">
                  <Plus className="mr-2 h-4 w-4" />
                  Add User
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Created</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {users.map((u) => (
                    <TableRow key={u.email}>
                      <TableCell className="font-medium">{u.name}</TableCell>
                      <TableCell>{u.email}</TableCell>
                      <TableCell>
                        <Badge variant="secondary" className="capitalize">
                          {u.role}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground text-xs">
                        {new Date(u.created * 1000).toLocaleDateString()}
                      </TableCell>
                    </TableRow>
                  ))}
                  {users.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={4} className="text-center py-8 text-muted-foreground">
                        {user?.role === 'admin' ? 'No users found' : 'Admin access required to view users'}
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="about">
          <Card>
            <CardHeader>
              <CardTitle>About POLYMORPH-4 Lite</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-4">
                <div className="h-16 w-16 bg-primary/10 rounded-lg flex items-center justify-center">
                  <Server className="h-8 w-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-bold">POLYMORPH-4 Lite</h3>
                  <p className="text-muted-foreground">Version 2.0.0</p>
                </div>
              </div>
              <div className="text-sm text-muted-foreground space-y-2">
                <p>
                  Advanced analytical instrument automation platform designed for high-throughput screening and real-time process control.
                </p>
                <p>
                  &copy; 2025 Polymorph Systems. All rights reserved.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </motion.div>
  )
}