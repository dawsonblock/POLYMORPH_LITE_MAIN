import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Beaker, CheckCircle, AlertTriangle } from "lucide-react";

export default function DashboardPage() {
    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                            Active Runs
                        </CardTitle>
                        <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">3</div>
                        <p className="text-xs text-muted-foreground">
                            +1 from last hour
                        </p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                            Device Status
                        </CardTitle>
                        <Beaker className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">Online</div>
                        <p className="text-xs text-muted-foreground">
                            All systems nominal
                        </p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                            Completed Workflows
                        </CardTitle>
                        <CheckCircle className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">+12</div>
                        <p className="text-xs text-muted-foreground">
                            +19% from last month
                        </p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">
                            Active Alerts
                        </CardTitle>
                        <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">0</div>
                        <p className="text-xs text-muted-foreground">
                            No critical alerts
                        </p>
                    </CardContent>
                </Card>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                <Card className="col-span-4">
                    <CardHeader>
                        <CardTitle>Recent Activity</CardTitle>
                    </CardHeader>
                    <CardContent className="pl-2">
                        {/* Placeholder for Activity Graph */}
                        <div className="h-[200px] flex items-center justify-center text-muted-foreground">
                            Activity Graph Placeholder
                        </div>
                    </CardContent>
                </Card>
                <Card className="col-span-3">
                    <CardHeader>
                        <CardTitle>System Health</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {/* Placeholder for System Health */}
                        <div className="space-y-4">
                            <div className="flex items-center">
                                <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                                <span className="text-sm font-medium">API Server</span>
                                <span className="ml-auto text-sm text-muted-foreground">99.9%</span>
                            </div>
                            <div className="flex items-center">
                                <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                                <span className="text-sm font-medium">Database</span>
                                <span className="ml-auto text-sm text-muted-foreground">100%</span>
                            </div>
                            <div className="flex items-center">
                                <div className="w-2 h-2 rounded-full bg-green-500 mr-2"></div>
                                <span className="text-sm font-medium">AI Service</span>
                                <span className="ml-auto text-sm text-muted-foreground">98.5%</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
