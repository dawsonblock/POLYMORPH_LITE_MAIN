import React, { useState, useEffect } from 'react';
import {
    Box,
    Container,
    Typography,
    Stepper,
    Step,
    StepLabel,
    Button,
    Paper,
    Card,
    CardContent,
    Grid,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Alert,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions
} from '@mui/material';
import { PlayArrow, Stop, CheckCircle } from '@mui/icons-material';
import api from '../api';

const steps = ['Hardware Profile', 'Select Workflow', 'Set Parameters', 'Execute', 'Review & Sign'];

interface WorkflowDef {
    id: string;
    name: string;
    description: string;
    estimated_duration_minutes: number;
}

export default function OperatorWizard() {
    const [activeStep, setActiveStep] = useState(0);
    const [hardwareProfile, setHardwareProfile] = useState('tier1');
    const [workflows, setWorkflows] = useState<WorkflowDef[]>([]);
    const [selectedWorkflow, setSelectedWorkflow] = useState<string>('');
    const [parameters, setParameters] = useState<Record<string, any>>({});
    const [running, setRunning] = useState(false);
    const [runId, setRunId] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const [signatureDialog, setSignatureDialog] = useState(false);
    const [signature, setSignature] = useState({ username: '', password: '', reason: '' });
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadWorkflows();
    }, [hardwareProfile]);

    const loadWorkflows = async () => {
        try {
            const res = await api.get(`/api/workflows?profile=${hardwareProfile}`);
            const workflowList = res.data || [];
            setWorkflows(workflowList);

            // Auto-select tier1 workflow if available
            const tier1Workflow = workflowList.find((w: WorkflowDef) =>
                w.id === 'tier1_daq_raman_sweep'
            );
            if (tier1Workflow) {
                setSelectedWorkflow(tier1Workflow.id);
            }
        } catch (err) {
            console.error('Failed to load workflows:', err);
        }
    };

    const handleNext = () => {
        setActiveStep((prev) => prev + 1);
        setError(null);
    };

    const handleBack = () => {
        setActiveStep((prev) => prev - 1);
        setError(null);
    };

    const handleExecute = async () => {
        setRunning(true);
        setError(null);

        try {
            const response = await api.post('/api/workflows/execute', {
                workflow_id: selectedWorkflow,
                parameters,
                operator_email: localStorage.getItem('user_email') || 'operator@example.com'
            });

            setRunId(response.data.run_id);

            // Simulate progress (in real app, would poll status)
            const interval = setInterval(() => {
                setProgress((prev) => {
                    if (prev >= 100) {
                        clearInterval(interval);
                        setRunning(false);
                        handleNext();
                        return 100;
                    }
                    return prev + 10;
                });
            }, 1000);

        } catch (err: any) {
            setError(err.response?.data?.detail || 'Workflow execution failed');
            setRunning(false);
        }
    };

    const handleSign = async () => {
        try {
            await api.post('/api/compliance/sign', {
                entity_type: 'workflow_execution',
                entity_id: runId,
                username: signature.username,
                password: signature.password,
                reason: signature.reason
            });

            setSignatureDialog(false);
            alert('Workflow completed and signed successfully!');

            // Reset wizard
            setActiveStep(0);
            setParameters({});
            setRunId(null);
            setProgress(0);

        } catch (err: any) {
            setError(err.response?.data?.detail || 'Signature failed');
        }
    };

    const handleDownloadReport = async () => {
        try {
            const response = await api.get(`/api/workflows/${runId}/report`, {
                responseType: 'blob'
            });

            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `workflow_report_${runId}.pdf`);
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (err) {
            console.error('Failed to download report:', err);
        }
    };

    const renderStepContent = () => {
        switch (activeStep) {
            case 0:
                // Hardware Profile
                return (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Select Hardware Profile
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                            Choose the hardware configuration for this workflow
                        </Typography>
                        <FormControl fullWidth>
                            <InputLabel>Hardware Profile</InputLabel>
                            <Select
                                value={hardwareProfile}
                                onChange={(e) => setHardwareProfile(e.target.value)}
                                label="Hardware Profile"
                            >
                                <MenuItem value="tier1">
                                    Tier-1 (NI DAQ + Ocean Optics) - Recommended
                                </MenuItem>
                                <MenuItem value="simulation" disabled>
                                    Simulation Mode - For Testing Only
                                </MenuItem>
                            </Select>
                        </FormControl>
                        <Alert severity="info" sx={{ mt: 2 }}>
                            Tier-1 hardware provides the highest precision and is recommended for production use.
                        </Alert>
                    </Box>
                );

            case 1:
                // Select Workflow
                return (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Select Workflow
                        </Typography>
                        <Grid container spacing={2}>
                            {workflows.map((workflow) => (
                                <Grid item xs={12} key={workflow.id}>
                                    <Card
                                        sx={{
                                            cursor: 'pointer',
                                            border: selectedWorkflow === workflow.id ? 2 : 1,
                                            borderColor: selectedWorkflow === workflow.id ? 'primary.main' : 'grey.300'
                                        }}
                                        onClick={() => setSelectedWorkflow(workflow.id)}
                                    >
                                        <CardContent>
                                            <Typography variant="h6">{workflow.name}</Typography>
                                            <Typography variant="body2" color="text.secondary">
                                                {workflow.description}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                Estimated duration: {workflow.estimated_duration_minutes} minutes
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            ))}
                        </Grid>
                    </Box>
                );

            case 2:
                // Set Parameters
                return (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Configure Parameters
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                            Set workflow parameters within safe operating ranges
                        </Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    label="Voltage Start (V)"
                                    type="number"
                                    value={parameters.voltage_start || 0}
                                    onChange={(e) => setParameters({ ...parameters, voltage_start: parseFloat(e.target.value) })}
                                    inputProps={{ min: 0, max: 5, step: 0.1 }}
                                    helperText="Safe range: 0-5V"
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    label="Voltage Stop (V)"
                                    type="number"
                                    value={parameters.voltage_stop || 5}
                                    onChange={(e) => setParameters({ ...parameters, voltage_stop: parseFloat(e.target.value) })}
                                    inputProps={{ min: 0, max: 5, step: 0.1 }}
                                    helperText="Safe range: 0-5V"
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    label="Voltage Step (V)"
                                    type="number"
                                    value={parameters.voltage_step || 0.5}
                                    onChange={(e) => setParameters({ ...parameters, voltage_step: parseFloat(e.target.value) })}
                                    inputProps={{ min: 0.1, max: 1, step: 0.1 }}
                                    helperText="Safe range: 0.1-1V"
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    label="Raman Integration Time (ms)"
                                    type="number"
                                    value={parameters.integration_time || 100}
                                    onChange={(e) => setParameters({ ...parameters, integration_time: parseInt(e.target.value) })}
                                    inputProps={{ min: 10, max: 1000, step: 10 }}
                                    helperText="Safe range: 10-1000ms"
                                />
                            </Grid>
                        </Grid>
                        <Alert severity="warning" sx={{ mt: 2 }}>
                            Parameters are validated against safety limits. Invalid values will be rejected.
                        </Alert>
                    </Box>
                );

            case 3:
                // Execute
                return (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Workflow Execution
                        </Typography>
                        {!running && progress === 0 ? (
                            <Box textAlign="center" py={4}>
                                <Typography variant="body1" gutterBottom>
                                    Ready to execute workflow
                                </Typography>
                                <Button
                                    variant="contained"
                                    size="large"
                                    startIcon={<PlayArrow />}
                                    onClick={handleExecute}
                                    sx={{ mt: 2 }}
                                >
                                    Start Execution
                                </Button>
                            </Box>
                        ) : (
                            <Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                    <CircularProgress variant="determinate" value={progress} sx={{ mr: 2 }} />
                                    <Typography variant="body1">
                                        Progress: {progress}%
                                    </Typography>
                                </Box>
                                {running && (
                                    <Alert severity="info">
                                        Workflow is executing... Do not navigate away from this page.
                                    </Alert>
                                )}
                                {!running && progress === 100 && (
                                    <Alert severity="success">
                                        Workflow completed successfully!
                                    </Alert>
                                )}
                            </Box>
                        )}
                    </Box>
                );

            case 4:
                // Review & Sign
                return (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Review & Sign
                        </Typography>
                        <Alert severity="success" sx={{ mb: 2 }}>
                            <Typography>Workflow completed successfully!</Typography>
                            <Typography variant="caption">Run ID: {runId}</Typography>
                        </Alert>
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <Button
                                    fullWidth
                                    variant="outlined"
                                    onClick={handleDownloadReport}
                                >
                                    Download Report (PDF/JSON)
                                </Button>
                            </Grid>
                            <Grid item xs={12}>
                                <Button
                                    fullWidth
                                    variant="contained"
                                    color="primary"
                                    startIcon={<CheckCircle />}
                                    onClick={() => setSignatureDialog(true)}
                                >
                                    Sign & Complete
                                </Button>
                            </Grid>
                        </Grid>
                    </Box>
                );

            default:
                return null;
        }
    };

    return (
        <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom>
                Operator Wizard
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Guided workflow execution with safety guardrails
            </Typography>

            <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
                {steps.map((label) => (
                    <Step key={label}>
                        <StepLabel>{label}</StepLabel>
                    </Step>
                ))}
            </Stepper>

            <Paper sx={{ p: 3, mb: 3 }}>
                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}
                {renderStepContent()}
            </Paper>

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Button
                    disabled={activeStep === 0 || running}
                    onClick={handleBack}
                >
                    Back
                </Button>
                <Button
                    variant="contained"
                    onClick={handleNext}
                    disabled={
                        activeStep === steps.length - 1 ||
                        (activeStep === 1 && !selectedWorkflow) ||
                        (activeStep === 3 && progress < 100) ||
                        running
                    }
                >
                    {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
                </Button>
            </Box>

            {/* Signature Dialog */}
            <Dialog open={signatureDialog} onClose={() => setSignatureDialog(false)}>
                <DialogTitle>Electronic Signature</DialogTitle>
                <DialogContent>
                    <TextField
                        fullWidth
                        label="Username"
                        margin="normal"
                        value={signature.username}
                        onChange={(e) => setSignature({ ...signature, username: e.target.value })}
                    />
                    <TextField
                        fullWidth
                        label="Password"
                        type="password"
                        margin="normal"
                        value={signature.password}
                        onChange={(e) => setSignature({ ...signature, password: e.target.value })}
                    />
                    <TextField
                        fullWidth
                        label="Signature Reason"
                        margin="normal"
                        multiline
                        rows={2}
                        value={signature.reason}
                        onChange={(e) => setSignature({ ...signature, reason: e.target.value })}
                        helperText="Required for 21 CFR 11 compliance"
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setSignatureDialog(false)}>Cancel</Button>
                    <Button
                        onClick={handleSign}
                        variant="contained"
                        disabled={!signature.username || !signature.password || !signature.reason}
                    >
                        Sign
                    </Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
}
