import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Paper, Typography } from '@mui/material';

interface DataPoint {
    x: number;
    y: number;
}

interface LiveChartProps {
    data: DataPoint[];
    title: string;
    color?: string;
}

const LiveChart: React.FC<LiveChartProps> = ({ data, title, color = "#8884d8" }) => {
    return (
        <Paper sx={{ p: 2, height: 300 }}>
            <Typography variant="h6" gutterBottom>
                {title}
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" type="number" domain={['auto', 'auto']} />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="y" stroke={color} dot={false} isAnimationActive={false} />
                </LineChart>
            </ResponsiveContainer>
        </Paper>
    );
};

export default LiveChart;
