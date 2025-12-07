"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface Snapshot {
    id: string;
    timestamp: string;
    trigger: string;
    workflow_execution_id?: string;
    occupancy?: number;
    risk?: number;
    age?: number;
}

export default function MemoryPage() {
    const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // TODO: Fetch from API
        setLoading(false);
        // Mock data for now
        setSnapshots([
            {
                id: "snap-001",
                timestamp: new Date().toISOString(),
                trigger: "workflow_end",
                workflow_execution_id: "wf-123",
                occupancy: 0.85,
                risk: 0.12,
                age: 150,
            },
        ]);
    }, []);

    const handleExport = async () => {
        try {
            const response = await fetch("/api/ai/export_memory", { method: "GET" });
            const data = await response.json();
            alert(`Memory exported to: ${data.path}`);
        } catch (err) {
            console.error("Export failed:", err);
        }
    };

    const handleReset = async () => {
        if (!confirm("Are you sure you want to reset the AI memory?")) return;
        try {
            await fetch("/api/ai/reset_memory", { method: "POST" });
            alert("Memory reset successfully");
        } catch (err) {
            console.error("Reset failed:", err);
        }
    };

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold">AI Memory Timeline</h1>
                    <p className="text-muted-foreground">
                        Memory snapshots and state management
                    </p>
                </div>
                <div className="flex gap-2">
                    <Button onClick={handleExport} variant="outline">
                        Export Memory
                    </Button>
                    <Button onClick={handleReset} variant="destructive">
                        Reset Memory
                    </Button>
                </div>
            </div>

            {/* Timeline */}
            <Card>
                <CardHeader>
                    <CardTitle>Snapshots</CardTitle>
                    <CardDescription>
                        Point-in-time captures of AI memory state
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {snapshots.length === 0 ? (
                        <p className="text-muted-foreground">No snapshots available</p>
                    ) : (
                        <div className="relative">
                            <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-border" />
                            <div className="space-y-6">
                                {snapshots.map((snap) => (
                                    <div key={snap.id} className="relative pl-10">
                                        <div className="absolute left-2.5 top-2 w-3 h-3 rounded-full bg-primary" />
                                        <Card>
                                            <CardContent className="pt-4">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <div className="flex items-center gap-2">
                                                            <Badge>{snap.trigger}</Badge>
                                                            {snap.workflow_execution_id && (
                                                                <span className="text-xs text-muted-foreground">
                                                                    {snap.workflow_execution_id}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <p className="text-sm text-muted-foreground mt-1">
                                                            {new Date(snap.timestamp).toLocaleString()}
                                                        </p>
                                                    </div>
                                                    <div className="text-right text-sm">
                                                        <p>Occupancy: {((snap.occupancy || 0) * 100).toFixed(1)}%</p>
                                                        <p>Risk: {snap.risk?.toFixed(3)}</p>
                                                        <p>Age: {snap.age}</p>
                                                    </div>
                                                </div>
                                            </CardContent>
                                        </Card>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Memory Operations */}
            <Card>
                <CardHeader>
                    <CardTitle>Memory Operations</CardTitle>
                    <CardDescription>
                        Import, export, and manage AI memory state
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="border rounded-lg p-4">
                            <h3 className="font-semibold mb-2">Import Memory</h3>
                            <p className="text-sm text-muted-foreground mb-3">
                                Restore from a previous checkpoint
                            </p>
                            <input
                                type="file"
                                accept=".npz"
                                className="block w-full text-sm text-muted-foreground
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-primary file:text-primary-foreground
                  hover:file:bg-primary/90"
                            />
                        </div>

                        <div className="border rounded-lg p-4">
                            <h3 className="font-semibold mb-2">Export Memory</h3>
                            <p className="text-sm text-muted-foreground mb-3">
                                Save current state as checkpoint
                            </p>
                            <Button onClick={handleExport} className="w-full">
                                Download Checkpoint
                            </Button>
                        </div>

                        <div className="border rounded-lg p-4">
                            <h3 className="font-semibold mb-2">Reset Memory</h3>
                            <p className="text-sm text-muted-foreground mb-3">
                                Clear all learned modes (irreversible)
                            </p>
                            <Button
                                onClick={handleReset}
                                variant="destructive"
                                className="w-full"
                            >
                                Reset to Initial State
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
