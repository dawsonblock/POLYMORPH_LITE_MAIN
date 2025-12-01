"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { api, endpoints } from "@/lib/api"
import { SpectralViewer } from "@/components/spectral-viewer"

export default function DevicesPage() {
    const [devices, setDevices] = useState<any[]>([])

    useEffect(() => {
        const fetchDevices = async () => {
            try {
                const res = await api.get(endpoints.devices.status)
                const data = res.data
                const mappedDevices = [
                    {
                        id: 'raman',
                        name: 'Ocean Optics USB2000+',
                        type: 'Spectrometer',
                        status: data.raman?.status === 'ok' ? 'online' : 'error',
                        lastCalibrated: '2023-10-20'
                    },
                    {
                        id: 'daq',
                        name: 'Red Pitaya STEMlab',
                        type: 'DAQ / Oscilloscope',
                        status: data.daq?.status === 'ok' ? 'online' : 'error'
                    },
                    {
                        id: 'balance',
                        name: 'Mettler Toledo Balance',
                        type: 'Scale',
                        status: 'offline'
                    }
                ]
                setDevices(mappedDevices)
            } catch (err) {
                console.error("Failed to fetch devices", err)
                setDevices([
                    { id: 'raman', name: 'Ocean Optics USB2000+', type: 'Spectrometer', status: 'error' },
                    { id: 'daq', name: 'Red Pitaya STEMlab', type: 'DAQ / Oscilloscope', status: 'error' }
                ])
            }
        }
        fetchDevices()
    }, [])

    const handleSelfTest = async (device: string) => {
        console.log(`Running self-test for ${device}`)
        try {
            await api.post(endpoints.devices.selfTest(device))
            alert(`Self-test initiated for ${device}`)
        } catch (err) {
            console.error("Self-test failed", err)
            alert("Failed to initiate self-test")
        }
    }

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <h2 className="text-3xl font-bold tracking-tight">Device Manager</h2>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {devices.map((dev) => (
                    <Card key={dev.id}>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <div className="space-y-1">
                                <CardTitle className="text-base font-medium">{dev.name}</CardTitle>
                                <CardDescription>{dev.type}</CardDescription>
                            </div>
                            <Badge variant={dev.status === 'online' ? 'default' : 'destructive'}>
                                {dev.status.toUpperCase()}
                            </Badge>
                        </CardHeader>
                        <CardContent>
                            <div className="text-sm text-muted-foreground mt-2">
                                Calibration: <span className={dev.lastCalibrated ? "text-foreground" : "text-destructive"}>
                                    {dev.lastCalibrated || "Required"}
                                </span>
                            </div>
                        </CardContent>
                        <CardFooter>
                            <Button onClick={() => handleSelfTest(dev.id)} variant="outline" className="w-full">
                                Run Self-Test
                            </Button>
                        </CardFooter>
                    </Card>
                ))}
            </div>

            <div className="grid gap-4 grid-cols-1 mt-4">
                <SpectralViewer />
            </div>
        </div>
    )
}
