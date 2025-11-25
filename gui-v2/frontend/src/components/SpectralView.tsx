import { useMemo } from 'react'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity } from 'lucide-react'
import useSystemStore from '@/stores/system-store'

export function SpectralView() {
    const { spectralData } = useSystemStore()

    // Transform data for Recharts
    // spectralData has { wavelengths: number[], intensities: number[] }
    // Recharts needs [{ wavelength: x, intensity: y }, ...]
    const chartData = useMemo(() => {
        if (!spectralData) return []

        return spectralData.wavelengths.map((wl, i) => ({
            wavelength: wl,
            intensity: spectralData.intensities[i],
        }))
    }, [spectralData])

    return (
        <Card className="col-span-full">
            <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                    <Activity className="h-5 w-5" />
                    <span>Real-time Spectrum</span>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="h-[300px] w-full">
                    {chartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                <XAxis
                                    dataKey="wavelength"
                                    label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }}
                                    type="number"
                                    domain={['auto', 'auto']}
                                />
                                <YAxis
                                    label={{ value: 'Intensity (a.u.)', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip
                                    labelFormatter={(value) => `${Number(value).toFixed(1)} nm`}
                                    formatter={(value: number) => [value.toFixed(1), 'Intensity']}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="intensity"
                                    stroke="#2563eb"
                                    strokeWidth={2}
                                    dot={false}
                                    isAnimationActive={false} // Disable animation for performance
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="flex h-full items-center justify-center text-muted-foreground">
                            Waiting for spectral data...
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
