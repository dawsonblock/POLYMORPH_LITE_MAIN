"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { api, endpoints } from "@/lib/api"

export default function AuditLogPage() {
    const [logs, setLogs] = useState<any[]>([])

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const res = await api.get(endpoints.audit.list)
                setLogs(res.data)
            } catch (err) {
                console.error("Failed to fetch audit logs", err)
            }
        }
        fetchLogs()
    }, [])

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight">Audit Trail</h2>
                <div className="flex items-center space-x-2">
                    <Button variant="outline">Export CSV</Button>
                </div>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>21 CFR Part 11 Audit Log</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="relative w-full overflow-auto">
                        <table className="w-full caption-bottom text-sm">
                            <thead className="[&_tr]:border-b">
                                <tr className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                                    <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Timestamp</th>
                                    <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">User</th>
                                    <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Action</th>
                                    <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Hash (SHA-256)</th>
                                </tr>
                            </thead>
                            <tbody className="[&_tr:last-child]:border-0">
                                {logs.map((log) => (
                                    <tr key={log.id} className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                                        <td className="p-4 align-middle">{new Date(log.ts * 1000).toISOString()}</td>
                                        <td className="p-4 align-middle">{log.actor}</td>
                                        <td className="p-4 align-middle font-medium">{log.event}</td>
                                        <td className="p-4 align-middle font-mono text-xs text-muted-foreground">{log.hash}</td>
                                    </tr>
                                ))}
                                {logs.length === 0 && (
                                    <tr>
                                        <td colSpan={4} className="p-4 align-middle text-center text-muted-foreground">No logs found.</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
