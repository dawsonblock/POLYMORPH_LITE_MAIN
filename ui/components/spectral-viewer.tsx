"use client"

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { api } from "@/lib/api"

interface SpectrumData {
    wavelength: number
    intensity: number
}

export function SpectralViewer() {
    const [data, setData] = useState<SpectrumData[]>([])
    const [isLive, setIsLive] = useState(false)
    const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null)

    const fetchSpectrum = async () => {
        // In a real app, this would connect to the WebSocket or poll the API
        // For now, we'll simulate a fetch or use the device self-test endpoint if available
        // Or better, let's simulate a "live" spectrum locally for the demo if the backend isn't streaming yet

        // Simulating data for visualization polish
        const points = []
        for (let i = 400; i < 800; i += 2) {
            points.push({
                wavelength: i,
                intensity: Math.random() * 100 + 500 * Math.exp(-Math.pow(i - 532, 2) / 100) // Peak at 532nm
            })
        }
        setData(points)
    }

    const toggleLive = () => {
        if (isLive) {
            if (intervalId) clearInterval(intervalId)
            setIsLive(false)
        } else {
            const id = setInterval(fetchSpectrum, 100)
            setIntervalId(id)
            setIsLive(true)
        }
    }

    useEffect(() => {
        fetchSpectrum()
        return () => {
            if (intervalId) clearInterval(intervalId)
        }
    }, [])

    return (
        <Card className="col-span-4">
            <CardHeader>
                <CardTitle className="flex justify-between items-center">
                    <span>Live Spectral Data</span>
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
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
