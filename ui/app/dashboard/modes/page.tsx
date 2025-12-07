"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

interface ModeStats {
    n_active: number;
    max_modes: number;
    active_indices: number[];
    occupancy: number[];
    risk: number[];
    age: number[];
    poly_count: number;
}

interface PolymorpInfo {
    id: string;
    first_seen: string;
}

export default function ModesPage() {
    const [modeStats, setModeStats] = useState<ModeStats | null>(null);
    const [polyIds, setPolyIds] = useState<Record<string, PolymorpInfo>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchData = async () => {
        try {
            setLoading(true);
            const [modesRes, polysRes] = await Promise.all([
                api.get("/ai/modes"),
                api.get("/ai/poly_ids"),
            ]);
            setModeStats(modesRes.data);
            setPolyIds(polysRes.data.details || {});
            setError(null);
        } catch (err: any) {
            setError(err.message || "Failed to fetch data");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const getRiskColor = (risk: number) => {
        if (risk < 0.3) return "bg-green-500";
        if (risk < 0.6) return "bg-yellow-500";
        return "bg-red-500";
    };

    const getOccupancyWidth = (occupancy: number) => {
        return `${Math.min(100, Math.max(5, occupancy * 100))}%`;
    };

    if (loading && !modeStats) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
        );
    }

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold">Modes Dashboard</h1>
                    <p className="text-muted-foreground">
                        Real-time PMM mode statistics and polymorph tracking
                    </p>
                </div>
                <Button onClick={fetchData} variant="outline">
                    Refresh
                </Button>
            </div>

            {error && (
                <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-3 rounded">
                    {error}
                </div>
            )}

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground">
                            Active Modes
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">
                            {modeStats?.n_active || 0}
                            <span className="text-sm font-normal text-muted-foreground">
                                / {modeStats?.max_modes || 32}
                            </span>
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground">
                            Polymorphs Discovered
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold text-green-600">
                            {modeStats?.poly_count || 0}
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground">
                            Avg Occupancy
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">
                            {modeStats?.occupancy?.length
                                ? (
                                    (modeStats.occupancy.reduce((a, b) => a + b, 0) /
                                        modeStats.occupancy.length) *
                                    100
                                ).toFixed(1)
                                : 0}
                            %
                        </div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-muted-foreground">
                            Max Risk
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">
                            {modeStats?.risk?.length
                                ? Math.max(...modeStats.risk).toFixed(2)
                                : 0}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Active Modes Table */}
            <Card>
                <CardHeader>
                    <CardTitle>Active Modes</CardTitle>
                    <CardDescription>
                        Current PMM mode prototypes with statistics
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b">
                                    <th className="text-left py-2 px-3">Index</th>
                                    <th className="text-left py-2 px-3">Occupancy</th>
                                    <th className="text-left py-2 px-3">Risk</th>
                                    <th className="text-left py-2 px-3">Age</th>
                                </tr>
                            </thead>
                            <tbody>
                                {modeStats?.active_indices?.map((idx, i) => (
                                    <tr key={idx} className="border-b hover:bg-muted/50">
                                        <td className="py-2 px-3">
                                            <Badge variant="outline">Mode {idx}</Badge>
                                        </td>
                                        <td className="py-2 px-3">
                                            <div className="flex items-center gap-2">
                                                <div className="w-24 bg-muted rounded-full h-2">
                                                    <div
                                                        className="bg-primary h-2 rounded-full"
                                                        style={{
                                                            width: getOccupancyWidth(modeStats.occupancy[i]),
                                                        }}
                                                    />
                                                </div>
                                                <span className="text-sm">
                                                    {(modeStats.occupancy[i] * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="py-2 px-3">
                                            <div className="flex items-center gap-2">
                                                <div
                                                    className={`w-3 h-3 rounded-full ${getRiskColor(
                                                        modeStats.risk[i]
                                                    )}`}
                                                />
                                                <span>{modeStats.risk[i].toFixed(3)}</span>
                                            </div>
                                        </td>
                                        <td className="py-2 px-3">{modeStats.age[i]}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>

            {/* Polymorph IDs */}
            <Card>
                <CardHeader>
                    <CardTitle>Discovered Polymorphs</CardTitle>
                    <CardDescription>
                        Unique polymorphs identified by the AI
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {Object.keys(polyIds).length === 0 ? (
                        <p className="text-muted-foreground">No polymorphs discovered yet</p>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {Object.entries(polyIds).map(([hash, info]) => (
                                <div
                                    key={hash}
                                    className="border rounded-lg p-4 hover:bg-muted/50"
                                >
                                    <div className="flex items-center justify-between">
                                        <Badge variant="secondary">{hash.slice(0, 8)}...</Badge>
                                        <span className="text-xs text-muted-foreground">
                                            {new Date(info.first_seen).toLocaleDateString()}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Risk Heatmap Placeholder */}
            <Card>
                <CardHeader>
                    <CardTitle>Risk Heatmap</CardTitle>
                    <CardDescription>
                        Visual representation of mode risk levels
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-8 gap-1">
                        {Array.from({ length: modeStats?.max_modes || 32 }).map((_, i) => {
                            const isActive = modeStats?.active_indices?.includes(i);
                            const riskIdx = modeStats?.active_indices?.indexOf(i);
                            const risk =
                                riskIdx !== undefined && riskIdx >= 0
                                    ? modeStats?.risk[riskIdx] || 0
                                    : 0;

                            return (
                                <div
                                    key={i}
                                    className={`aspect-square rounded ${isActive
                                            ? getRiskColor(risk)
                                            : "bg-muted"
                                        } flex items-center justify-center text-xs font-medium ${isActive ? "text-white" : "text-muted-foreground"
                                        }`}
                                    title={`Mode ${i}${isActive ? ` (risk: ${risk.toFixed(3)})` : " (inactive)"}`}
                                >
                                    {i}
                                </div>
                            );
                        })}
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
