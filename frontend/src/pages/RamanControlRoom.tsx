import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Button,
    Grid,
    Alert,
    Chip,
    LinearProgress,
    IconButton,
    Paper,
} from '@mui/material';
import {
    PlayArrow,
    Stop,
    Warning as WarningIcon,
    CheckCircle,
    Error as ErrorIcon,
} from '@mui/icons-material';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
} from 'recharts';

interface SpectrumData {
    wavelengths: number[];
    intensities: number[];
    timestamp: string;
}

interface AIPrediction {
    modes: string[];
    polymorphs: string[];
    confidence: number;
    new_polymorph: boolean;
    predicted_finish?: string;
}

interface GatingState {
    rules: Array<{ name: string; status: string; value?: number }>;
    should_stop: boolean;
    message?: string;
}

interface WorkflowStatus {
    run_id: string;
    status: string;
    current_step?: number;
    total_steps?: number;
}

const RamanControlRoom: React.FC = () => {
    const [connected, setConnected] = useState(false);
    const [acquiring, setAcquiring] = useState(false);
    const [spectrum, setSpectrum] = useState<SpectrumData | null>(null);
    const [aiPrediction, setAIPrediction] = useState<AIPrediction | null>(null);
    const [gatingState, setGatingState] = useState<GatingState | null>(null);
    const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus | null>(null);
    const [error, setError] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);

    // WebSocket connection
    useEffect(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/raman/stream`;

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('WebSocket connected');
            setConnected(true);
            setError(null);
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            } catch (err) {
                console.error('Failed to parse WebSocket message:', err);
            }
        };

        ws.onerror = (event) => {
            console.error('WebSocket error:', event);
            setError('WebSocket connection error');
            setConnected(false);
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            setConnected(false);
        };

        // Cleanup on unmount
        return () => {
            ws.close();
        };
    }, []);

    const handleWebSocketMessage = (message: any) => {
        switch (message.type) {
            case 'spectrum':
                setSpectrum(message.data);
                break;
            case 'ai_prediction':
                setAIPrediction(message.data);
                break;
            case 'gating_state':
                setGatingState(message.data);
                break;
            case 'workflow_status':
                setWorkflowStatus(message.data);
                break;
            case 'acquisition_started':
                setAcquiring(true);
                break;
            case 'acquisition_stopped':
            case 'emergency_stop':
                setAcquiring(false);
                break;
            case 'error':
                setError(message.data.message);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    };

    const handleStartAcquisition = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'start_acquisition',
                data: { integration_time_ms: 20.0 }
            }));
        }
    };

    const handleStopAcquisition = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'stop_acquisition'
            }));
        }
    };

    const handleEmergencyStop = async () => {
        try {
            const response = await fetch('/api/raman/emergency-stop', {
                method: 'POST',
            });
            const data = await response.json();
            if (data.status === 'stopped') {
                setAcquiring(false);
            }
        } catch (err) {
            console.error('Emergency stop failed:', err);
            setError('Emergency stop failed');
        }
    };

    // Prepare chart data
    const chartData = spectrum
        ? spectrum.wavelengths.map((wavelength, index) => ({
            wavelength,
            intensity: spectrum.intensities[index],
        }))
        : [];

    return (
        <Box sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>
                Raman Control Room
            </Typography>

            {/* Connection Status */}
            <Box sx={{ mb: 2 }}>
                <Chip
                    icon={connected ? <CheckCircle /> : <ErrorIcon />}
                    label={connected ? 'Connected' : 'Disconnected'}
                    color={connected ? 'success' : 'error'}
                    sx={{ mr: 1 }}
                />
                <Chip
                    label={acquiring ? 'Acquiring' : 'Idle'}
                    color={acquiring ? 'primary' : 'default'}
                />
            </Box>

            {/* Error Alert */}
            {error && (
                <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Grid container spacing={3}>
                {/* Control Panel */}
                <Grid item xs={12} md={3}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Controls
                            </Typography>

                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                <Button
                                    variant="contained"
                                    color="primary"
                                    startIcon={<PlayArrow />}
                                    onClick={handleStartAcquisition}
                                    disabled={!connected || acquiring}
                                    fullWidth
                                >
                                    Start Acquisition
                                </Button>

                                <Button
                                    variant="outlined"
                                    color="secondary"
                                    startIcon={<Stop />}
                                    onClick={handleStopAcquisition}
                                    disabled={!connected || !acquiring}
                                    fullWidth
                                >
                                    Stop Acquisition
                                </Button>

                                <Button
                                    variant="contained"
                                    color="error"
                                    startIcon={<WarningIcon />}
                                    onClick={handleEmergencyStop}
                                    disabled={!connected}
                                    fullWidth
                                >
                                    Emergency Stop
                                </Button>
                            </Box>
                        </CardContent>
                    </Card>

                    {/* Workflow Status */}
                    {workflowStatus && (
                        <Card sx={{ mt: 2 }}>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Workflow Status
                                </Typography>
                                <Typography variant="body2">
                                    Run ID: {workflowStatus.run_id}
                                </Typography>
                                <Typography variant="body2">
                                    Status: {workflowStatus.status}
                                </Typography>
                                {workflowStatus.current_step !== undefined && (
                                    <>
                                        <Typography variant="body2" sx={{ mt: 1 }}>
                                            Step: {workflowStatus.current_step} / {workflowStatus.total_steps}
                                        </Typography>
                                        <LinearProgress
                                            variant="determinate"
                                            value={(workflowStatus.current_step / (workflowStatus.total_steps || 1)) * 100}
                                            sx={{ mt: 1 }}
                                        />
                                    </>
                                )}
                            </CardContent>
                        </Card>
                    )}
                </Grid>

                {/* Live Spectrum Chart */}
                <Grid item xs={12} md={9}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Live Raman Spectrum
                            </Typography>

                            {spectrum ? (
                                <ResponsiveContainer width="100%" height={400}>
                                    <LineChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis
                                            dataKey="wavelength"
                                            label={{ value: 'Wavelength (nm)', position: 'insideBottom', offset: -5 }}
                                        />
                                        <YAxis
                                            label={{ value: 'Intensity', angle: -90, position: 'insideLeft' }}
                                        />
                                        <Tooltip />
                                        <Legend />
                                        <Line
                                            type="monotone"
                                            dataKey="intensity"
                                            stroke="#8884d8"
                                            dot={false}
                                            isAnimationActive={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            ) : (
                                <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Typography color="text.secondary">
                                        No spectrum data available. Start acquisition to view live data.
                                    </Typography>
                                </Box>
                            )}

                            {spectrum && (
                                <Typography variant="caption" color="text.secondary">
                                    Last updated: {new Date(spectrum.timestamp).toLocaleTimeString()}
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* AI Predictions */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                AI Predictions (PMM Brain)
                            </Typography>

                            {aiPrediction ? (
                                <Box>
                                    <Typography variant="body2" gutterBottom>
                                        <strong>Modes:</strong> {aiPrediction.modes.join(', ') || 'None detected'}
                                    </Typography>
                                    <Typography variant="body2" gutterBottom>
                                        <strong>Polymorphs:</strong> {aiPrediction.polymorphs.join(', ') || 'None detected'}
                                    </Typography>
                                    <Typography variant="body2" gutterBottom>
                                        <strong>Confidence:</strong> {(aiPrediction.confidence * 100).toFixed(1)}%
                                    </Typography>

                                    {aiPrediction.new_polymorph && (
                                        <Alert severity="warning" sx={{ mt: 2 }}>
                                            🎉 New polymorph detected!
                                        </Alert>
                                    )}

                                    {aiPrediction.predicted_finish && (
                                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                            Predicted finish: {aiPrediction.predicted_finish}
                                        </Typography>
                                    )}
                                </Box>
                            ) : (
                                <Typography color="text.secondary">
                                    No AI predictions available yet.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>

                {/* Gating Engine State */}
                <Grid item xs={12} md={6}>
                    <Card>
                        <CardContent>
                            <Typography variant="h6" gutterBottom>
                                Gating Engine
                            </Typography>

                            {gatingState ? (
                                <Box>
                                    {gatingState.rules.map((rule, index) => (
                                        <Paper key={index} sx={{ p: 1, mb: 1 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <Typography variant="body2">
                                                    {rule.name}
                                                </Typography>
                                                <Chip
                                                    label={rule.status}
                                                    color={rule.status === 'pass' ? 'success' : 'warning'}
                                                    size="small"
                                                />
                                            </Box>
                                            {rule.value !== undefined && (
                                                <Typography variant="caption" color="text.secondary">
                                                    Value: {rule.value}
                                                </Typography>
                                            )}
                                        </Paper>
                                    ))}

                                    {gatingState.should_stop && (
                                        <Alert severity="error" sx={{ mt: 2 }}>
                                            ⚠️ Gating condition triggered: {gatingState.message || 'Stop condition met'}
                                        </Alert>
                                    )}
                                </Box>
                            ) : (
                                <Typography color="text.secondary">
                                    No gating state available.
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>
        </Box>
    );
};

export default RamanControlRoom;
