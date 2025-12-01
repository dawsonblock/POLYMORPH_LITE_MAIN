"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Play, Pause, Square, Loader2 } from "lucide-react"
import { api, endpoints } from "@/lib/api"

interface WorkflowStep {
    id: string
    name: string
    status: 'pending' | 'running' | 'completed' | 'failed'
}

export default function WorkflowsPage() {
    const [activeWorkflow, setActiveWorkflow] = useState<string | null>(null)
    const [runId, setRunId] = useState<string | null>(null)
    const [steps, setSteps] = useState<WorkflowStep[]>([])
    const [status, setStatus] = useState('idle')
    const [logs, setLogs] = useState<string[]>([])
    const [availableWorkflows, setAvailableWorkflows] = useState<any[]>([])

    // Fetch available workflows on mount
    useEffect(() => {
        const fetchWorkflows = async () => {
            try {
                const res = await api.get(endpoints.workflows.list)
                setAvailableWorkflows(res.data)
            } catch (err) {
                console.error("Failed to fetch workflows", err)
                // Fallback for demo
                setAvailableWorkflows([{ id: 'wf-demo', name: 'Demo Workflow (Fallback)' }])
            }
        }
        fetchWorkflows()
    }, [])

    // Poll for status when running
    useEffect(() => {
        if (!runId) return

        const interval = setInterval(async () => {
            try {
                const res = await api.get(endpoints.workflows.get(runId))
                const state = res.data
                setStatus(state.status)

                if (state.steps) {
                    setSteps(state.steps.map((s: any) => ({
                        id: s.id,
                        name: s.name,
                        status: s.status
                    })))
                }

                if (state.last_log) {
                    setLogs(prev => [...prev, `[${new Date().toISOString()}] ${state.last_log}`])
                }
            } catch (err) {
                console.error("Poll error", err)
            }
        }, 1000)

        return () => clearInterval(interval)
    }, [runId])

    const handleStart = async () => {
        try {
            const wfId = availableWorkflows[0]?.id || 'wf-123'
            const res = await api.post(endpoints.workflows.run, {
                workflow_version_id: wfId,
                context: {}
            })
            setRunId(res.data.run_id)
            setActiveWorkflow(wfId)
            setStatus('running')
            setLogs(prev => [...prev, `[${new Date().toISOString()}] Workflow started (Run ID: ${res.data.run_id})`])
        } catch (err) {
            console.error("Failed to start workflow", err)
            alert("Failed to start workflow. Ensure backend is running.")
        }
    }

    const handlePause = async () => {
        if (!runId) return
        await api.post(endpoints.workflows.pause(runId))
        setStatus('paused')
    }

    const handleResume = async () => {
        if (!runId) return
        await api.post(endpoints.workflows.resume(runId))
        setStatus('running')
    }

    const handleStop = async () => {
        if (!runId) return
        await api.post(endpoints.workflows.cancel(runId))
        setStatus('idle')
        setActiveWorkflow(null)
        setRunId(null)
        setSteps([])
    }

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight">Workflow Runner</h2>
            </div>

            <div className="grid gap-4 grid-cols-1 lg:grid-cols-3">
                {/* Left Panel: Controls & Steps */}
                <div className="lg:col-span-2 space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Control Panel</CardTitle>
                            <CardDescription>Manage workflow execution</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-4 mb-6">
                                {!runId ? (
                                    <Button onClick={handleStart} className="w-full sm:w-auto">
                                        <Play className="mr-2 h-4 w-4" /> Start Workflow
                                    </Button>
                                ) : (
                                    <>
                                        {status === 'running' ? (
                                            <Button onClick={handlePause} variant="secondary">
                                                <Pause className="mr-2 h-4 w-4" /> Pause
                                            </Button>
                                        ) : (
                                            <Button onClick={handleResume} className="bg-green-600 hover:bg-green-700">
                                                <Play className="mr-2 h-4 w-4" /> Resume
                                            </Button>
                                        )}
                                        <Button onClick={handleStop} variant="destructive">
                                            <Square className="mr-2 h-4 w-4" /> Stop
                                        </Button>
                                    </>
                                )}
                            </div>

                            <div className="space-y-3">
                                {steps.map((step, idx) => (
                                    <div key={step.id} className={`p-4 rounded-lg border flex items-center justify-between ${step.status === 'running' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-border'}`}>
                                        <div className="flex items-center gap-3">
                                            <span className="w-6 h-6 rounded-full bg-secondary flex items-center justify-center text-xs font-bold">
                                                {idx + 1}
                                            </span>
                                            <span className="font-medium">{step.name}</span>
                                        </div>
                                        <span className="text-sm text-muted-foreground capitalize flex items-center gap-2">
                                            {step.status === 'running' && <Loader2 className="h-3 w-3 animate-spin" />}
                                            {step.status}
                                        </span>
                                    </div>
                                ))}
                                {steps.length === 0 && <p className="text-muted-foreground italic text-center py-8">No workflow active.</p>}
                            </div>
                        </CardContent>
                    </Card>

                    {/* Live Telemetry Placeholder */}
                    <Card>
                        <CardHeader>
                            <CardTitle>Live Telemetry</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="h-64 flex items-center justify-center border-2 border-dashed border-border rounded-lg bg-muted/50">
                                <p className="text-muted-foreground">Live Telemetry Visualization</p>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Right Panel: Logs */}
                <Card className="h-[600px] flex flex-col">
                    <CardHeader>
                        <CardTitle>System Logs</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 overflow-hidden">
                        <div className="h-full overflow-y-auto font-mono text-xs space-y-1 bg-slate-950 text-slate-50 p-4 rounded-md">
                            {logs.map((log, i) => (
                                <div key={i} className="break-all border-b border-slate-800 pb-1 mb-1 last:border-0">{log}</div>
                            ))}
                            {logs.length === 0 && <span className="text-slate-500">Waiting for logs...</span>}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
