"use client"

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { io, Socket } from "socket.io-client"

interface SpectrumData {
    wavelength: number
    intensity: number
}

export function SpectralViewer() {
    const [data, setData] = useState<SpectrumData[]>([])
    const [isLive, setIsLive] = useState(false)
    const [socket, setSocket] = useState<Socket | null>(null)

    useEffect(() => {
        // Initialize Socket.IO connection
        // Note: In Next.js, we might need to ensure this only runs on client
        const socketInstance = io(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001", {
            path: "/socket.io",
            transports: ["websocket"],
            autoConnect: false
        })

        socketInstance.on("connect", () => {
            console.log("SpectralViewer connected to WebSocket")
        })

        socketInstance.on("spectral_data", (payload: any) => {
            // Payload expected: { wavelengths: [], intensities: [], ... }
            if (payload.wavelengths && payload.intensities) {
                const points = payload.wavelengths.map((wl: number, idx: number) => ({
                    wavelength: wl,
                    intensity: payload.intensities[idx]
                }))
                setData(points)
            }
        })

        setSocket(socketInstance)

        return () => {
            socketInstance.disconnect()
        }
    }, [])

    const toggleLive = () => {
        if (!socket) return

        if (isLive) {
            socket.disconnect()
            setIsLive(false)
        } else {
            socket.connect()
            setIsLive(true)
        }
    }

    return (
        <Card className="col-span-4">
            <CardHeader>
                <CardTitle className="flex justify-between items-center">
                    <span>Live Spectral Data (WebSocket)</span>
                    <Button
                        variant={isLive ? "destructive" : "default"}
                        onClick={toggleLive}
                    >
                        {isLive ? "Stop Acquisition" : "Start Live View"}
                    </Button>
                </CardTitle>
            </CardHeader>
            <CardContent className="pl-2">
                <div className="h-[350px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis
                                dataKey="wavelength"
                                stroke="#888888"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                                label={{ value: 'Wavelength (nm)', position: 'insideBottomRight', offset: -5 }}
                            />
                            <YAxis
                                stroke="#888888"
                                fontSize={12}
                                tickLine={false}
                                axisLine={false}
                                label={{ value: 'Intensity (a.u.)', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                                labelStyle={{ color: 'hsl(var(--foreground))' }}
                            />
                            <Line
                                type="monotone"
                                dataKey="intensity"
                                stroke="hsl(var(--primary))"
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false} // Disable animation for performance
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
