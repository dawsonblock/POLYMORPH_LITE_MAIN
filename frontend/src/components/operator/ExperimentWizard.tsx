import React, { useState, useEffect } from 'react';
import {
    Box,
    Stepper,
    Step,
    StepLabel,
    Button,
    Typography,
    Paper,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Checkbox,
    FormControlLabel,
    TextField,
    Grid,
    Alert
} from '@mui/material';
import LiveChart from './LiveChart';

const steps = ['Select Workflow', 'Configure Hardware', 'Review Parameters', 'Run Experiment', 'Report'];

const ExperimentWizard: React.FC = () => {
    const [activeStep, setActiveStep] = useState(0);
    const [workflow, setWorkflow] = useState('');
    const [hardwareReady, setHardwareReady] = useState({ daq: false, raman: false });
    const [isRunning, setIsRunning] = useState(false);
    const [progress, setProgress] = useState(0);
    const [chartData, setChartData] = useState<{ x: number; y: number }[]>([]);

    // Mock data simulation
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (isRunning && activeStep === 3) {
            interval = setInterval(() => {
                setProgress((prev) => {
                    if (prev >= 100) {
                        setIsRunning(false);
                        clearInterval(interval);
                        return 100;
                    }
                    return prev + 1;
                });

                setChartData((prev) => {
                    const x = prev.length;
                    const y = Math.sin(x * 0.1) * 10 + Math.random() * 2;
                    return [...prev, { x, y }].slice(-100); // Keep last 100 points
                });
            }, 100);
        }
        return () => clearInterval(interval);
    }, [isRunning, activeStep]);

    const handleNext = () => {
        setActiveStep((prev) => prev + 1);
    };

    const handleBack = () => {
        setActiveStep((prev) => prev - 1);
    };

    const handleRun = () => {
        setIsRunning(true);
        setChartData([]);
        setProgress(0);
    };

    const handleDownloadReport = () => {
        alert("Downloading PDF Report... (Mock)");
        // In real implementation, use jspdf here
    };

    const renderStepContent = (step: number) => {
        switch (step) {
            case 0:
                return (
                    <Box sx={{ mt: 2 }}>
                        <FormControl fullWidth>
                            <InputLabel>Workflow</InputLabel>
                            <Select
                                value={workflow}
                                label="Workflow"
                                onChange={(e) => setWorkflow(e.target.value)}
                            >
                                <MenuItem value="hero_crystallization">Hero Crystallization</MenuItem>
                                <MenuItem value="calibration">System Calibration</MenuItem>
                                <MenuItem value="hardware_test">Hardware Test</MenuItem>
                            </Select>
                        </FormControl>
                    </Box>
                );
            case 1:
                return (
                    <Box sx={{ mt: 2 }}>
                        <FormControlLabel
                            control={<Checkbox checked={hardwareReady.daq} onChange={(e) => setHardwareReady({ ...hardwareReady, daq: e.target.checked })} />}
                            label="NI DAQ Connected"
                        />
                        <FormControlLabel
                            control={<Checkbox checked={hardwareReady.raman} onChange={(e) => setHardwareReady({ ...hardwareReady, raman: e.target.checked })} />}
                            label="Ocean Optics Spectrometer Connected"
                        />
                    </Box>
                );
            case 2:
                return (
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="h6">Review Parameters</Typography>
                        <Grid container spacing={2}>
                            <Grid size={{ xs: 6 }}>
                                <TextField fullWidth label="Laser Power (mW)" defaultValue="100" disabled />
                            </Grid>
                            <Grid size={{ xs: 6 }}>
                                <TextField fullWidth label="Integration Time (ms)" defaultValue="1000" disabled />
                            </Grid>
                        </Grid>
                        <Alert severity="info" sx={{ mt: 2 }}>
                            Workflow: {workflow || "None Selected"}
                        </Alert>
                    </Box>
                );
            case 3:
                return (
                    <Box sx={{ mt: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Experiment Progress: {progress}%
                        </Typography>
                        <Box sx={{ width: '100%', mb: 2 }}>
                            <div style={{ width: `${progress}%`, height: 10, backgroundColor: '#4caf50', transition: 'width 0.1s' }} />
                        </Box>

                        <LiveChart data={chartData} title="Real-time Raman Spectrum" />

                        {!isRunning && progress === 100 && (
                            <Alert severity="success" sx={{ mt: 2 }}>Experiment Complete!</Alert>
                        )}
                        {!isRunning && progress === 0 && (
                            <Button variant="contained" color="primary" onClick={handleRun}>
                                Start Execution
                            </Button>
                        )}
                    </Box>
                );
            case 4:
                return (
                    <Box sx={{ mt: 2, textAlign: 'center' }}>
                        <Typography variant="h5" gutterBottom>Experiment Complete</Typography>
                        <Typography variant="body1">
                            Workflow: {workflow}<br />
                            Status: Success<br />
                            Data Points: {chartData.length}
                        </Typography>
                        <Button variant="contained" color="secondary" onClick={handleDownloadReport} sx={{ mt: 2 }}>
                            Download Signed PDF Report
                        </Button>
                    </Box>
                );
            default:
                return "Unknown step";
        }
    };

    return (
        <Paper sx={{ p: 3 }}>
            <Typography variant="h4" gutterBottom>Operator Wizard</Typography>
            <Stepper activeStep={activeStep}>
                {steps.map((label) => (
                    <Step key={label}>
                        <StepLabel>{label}</StepLabel>
                    </Step>
                ))}
            </Stepper>
            <Box sx={{ mt: 4, mb: 4, minHeight: 300 }}>
                {renderStepContent(activeStep)}
            </Box>
            <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
                <Button
                    color="inherit"
                    disabled={activeStep === 0}
                    onClick={handleBack}
                    sx={{ mr: 1 }}
                >
                    Back
                </Button>
                <Box sx={{ flex: '1 1 auto' }} />
                {activeStep < steps.length - 1 && activeStep !== 3 && (
                    <Button onClick={handleNext}>
                        Next
                    </Button>
                )}
                {activeStep === 3 && !isRunning && progress === 100 && (
                    <Button onClick={handleNext}>
                        Next
                    </Button>
                )}
            </Box>
        </Paper>
    );
};

export default ExperimentWizard;
