import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Upload, FileText, Loader2 } from 'lucide-react'
import { toast } from 'sonner'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'

import useAuthStore from '@/stores/auth-store'

interface Workflow {
  id: string
  name: string
  entry_step: string
  num_steps: number
}

export function RecipeManager() {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [executingId, setExecutingId] = useState<string | null>(null)
  const { token } = useAuthStore()

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/workflows/', {
        headers: { Authorization: `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to fetch workflows')
      const data = await response.json()
      setWorkflows(data)
    } catch (error) {
      console.error('Error fetching workflows:', error)
      toast.error('Failed to load workflows')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchWorkflows()
  }, [token])

  const handleRun = async (id: string, name: string) => {
    setExecutingId(id)
    toast.info(`Starting workflow: ${name}...`)
    try {
      const response = await fetch(`http://localhost:8000/api/workflows/${id}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({})
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.detail || 'Execution failed')
      }

      if (result.success) {
        toast.success(`Workflow '${name}' completed successfully!`, {
          description: `Duration: ${result.duration_seconds.toFixed(2)}s`
        })
      } else {
        toast.error(`Workflow '${name}' failed`, {
          description: result.error
        })
      }
    } catch (error) {
      console.error('Execution error:', error)
      toast.error(`Failed to execute workflow: ${error}`)
    } finally {
      setExecutingId(null)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = async (event) => {
      const content = event.target?.result as string
      try {
        const response = await fetch('http://localhost:8000/api/workflows/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify({ yaml_content: content })
        })

        if (!response.ok) {
          const err = await response.json()
          throw new Error(err.detail || 'Upload failed')
        }

        toast.success('Workflow uploaded successfully')
        fetchWorkflows()
      } catch (error) {
        toast.error(`Upload failed: ${error}`)
      }
    }
    reader.readAsText(file)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Workflow Manager</h2>
          <p className="text-muted-foreground">Manage and execute experimental recipes (workflows).</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" className="relative">
            <input
              type="file"
              accept=".yaml,.yml"
              className="absolute inset-0 opacity-0 cursor-pointer"
              onChange={handleFileUpload}
            />
            <Upload className="mr-2 h-4 w-4" />
            Upload YAML
          </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {isLoading ? (
          <div className="col-span-full flex justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : workflows.length === 0 ? (
          <div className="col-span-full text-center py-12 text-muted-foreground">
            No workflows found. Upload a YAML file to get started.
          </div>
        ) : (
          workflows.map((workflow) => (
            <Card key={workflow.id} className="flex flex-col">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-xl">{workflow.name}</CardTitle>
                    <CardDescription className="font-mono text-xs text-muted-foreground">
                      ID: {workflow.id}
                    </CardDescription>
                  </div>
                  <Badge variant="secondary">
                    {workflow.num_steps} Steps
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="flex-1 flex flex-col justify-end gap-4">
                <div className="text-sm text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    <span>Entry: <span className="font-mono">{workflow.entry_step}</span></span>
                  </div>
                </div>

                <Separator />

                <div className="flex items-center gap-2">
                  <Button
                    className="w-full"
                    onClick={() => handleRun(workflow.id, workflow.name)}
                    disabled={executingId !== null}
                  >
                    {executingId === workflow.id ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Run Workflow
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </motion.div>
  )
}