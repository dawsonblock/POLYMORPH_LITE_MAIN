import React, { useState, useEffect } from 'react';
import {
    Box,
    Container,
    Typography,
    Grid,
    Card,
    CardContent,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Chip,
    Button,
    TextField,
    MenuItem,
    IconButton
} from '@mui/material';
import { Download, Refresh, Search } from '@mui/icons-material';
import api from '../api';

interface PolymorphEvent {
    event_id: string;
    polymorph_name: string;
    polymorph_id: number;
    confidence: number;
    detected_at: number;
    operator: string;
    model_version: string;
}

interface PolymorphStats {
    total_detections: number;
    average_confidence: number;
    by_polymorph_type: Array<{ name: string; count: number }>;
}

export default function PolymorphExplorer() {
    const [events, setEvents] = useState<PolymorphEvent[]>([]);
    const [stats, setStats] = useState<PolymorphStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterType, setFilterType] = useState('all');

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        setLoading(true);
        try {
            // Load events
            const eventsRes = await api.get('/api/polymorph/events?limit=100');
            setEvents(eventsRes.data.events || []);

            // Load statistics
            const statsRes = await api.get('/api/polymorph/statistics');
            setStats(statsRes.data);
        } catch (error) {
            console.error('Failed to load polymorph data:', error);
        }
        setLoading(false);
    };

    const handleDownloadReport = async (eventId: string) => {
        try {
            const response = await api.post('/api/polymorph/report', {
                event_id: eventId,
                format: 'json',
                include_spectrum: true
            });

            // Download JSON report
            const blob = new Blob([JSON.stringify(response.data, null, 2)], {
                type: 'application/json'
            });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `polymorph_report_${eventId}.json`;
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Failed to generate report:', error);
        }
    };

    const filteredEvents = events.filter(event => {
        const matchesSearch = event.polymorph_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            event.event_id.includes(searchTerm);
        const matchesFilter = filterType === 'all' || event.polymorph_name === filterType;
        return matchesSearch && matchesFilter;
    });

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.9) return 'success';
        if (confidence >= 0.7) return 'warning';
        return 'error';
    };

    return (
        <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" gutterBottom>
                    Polymorph Explorer
                </Typography>
                <Typography variant="body1" color="text.secondary">
                    Browse detected polymorphs, view signatures, and download reports
                </Typography>
            </Box>

            {/* Statistics Cards */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={4}>
                    <Card>
                        <CardContent>
                            <Typography color="text.secondary" gutterBottom>
                                Total Detections
                            </Typography>
                            <Typography variant="h3">
                                {stats?.total_detections || 0}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                    <Card>
                        <CardContent>
                            <Typography color="text.secondary" gutterBottom>
                                Average Confidence
                            </Typography>
                            <Typography variant="h3">
                                {((stats?.average_confidence || 0) * 100).toFixed(1)}%
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} md={4}>
                    <Card>
                        <CardContent>
                            <Typography color="text.secondary" gutterBottom>
                                Unique Forms
                            </Typography>
                            <Typography variant="h3">
                                {stats?.by_polymorph_type.length || 0}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Filters */}
            <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
                <TextField
                    label="Search"
                    variant="outlined"
                    size="small"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                        startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                    }}
                    sx={{ flexGrow: 1 }}
                />
                <TextField
                    select
                    label="Filter by Type"
                    variant="outlined"
                    size="small"
                    value={filterType}
                    onChange={(e) => setFilterType(e.target.value)}
                    sx={{ minWidth: 200 }}
                >
                    <MenuItem value="all">All Types</MenuItem>
                    {stats?.by_polymorph_type.map(type => (
                        <MenuItem key={type.name} value={type.name}>
                            {type.name} ({type.count})
                        </MenuItem>
                    ))}
                </TextField>
                <IconButton onClick={loadData} color="primary">
                    <Refresh />
                </IconButton>
            </Box>

            {/* Events Table */}
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Event ID</TableCell>
                            <TableCell>Polymorph Type</TableCell>
                            <TableCell>Confidence</TableCell>
                            <TableCell>Detected At</TableCell>
                            <TableCell>Operator</TableCell>
                            <TableCell>Model Version</TableCell>
                            <TableCell align="right">Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {loading ? (
                            <TableRow>
                                <TableCell colSpan={7} align="center">
                                    Loading...
                                </TableCell>
                            </TableRow>
                        ) : filteredEvents.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={7} align="center">
                                    No polymorphs detected yet
                                </TableCell>
                            </TableRow>
                        ) : (
                            filteredEvents.map((event) => (
                                <TableRow key={event.event_id} hover>
                                    <TableCell>
                                        <code style={{ fontSize: '0.85em' }}>
                                            {event.event_id.substring(0, 8)}...
                                        </code>
                                    </TableCell>
                                    <TableCell>
                                        <Typography variant="body2" fontWeight="medium">
                                            {event.polymorph_name}
                                        </Typography>
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            label={`${(event.confidence * 100).toFixed(1)}%`}
                                            color={getConfidenceColor(event.confidence)}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>
                                        {new Date(event.detected_at * 1000).toLocaleString()}
                                    </TableCell>
                                    <TableCell>{event.operator}</TableCell>
                                    <TableCell>
                                        <Chip label={event.model_version} size="small" variant="outlined" />
                                    </TableCell>
                                    <TableCell align="right">
                                        <Button
                                            size="small"
                                            startIcon={<Download />}
                                            onClick={() => handleDownloadReport(event.event_id)}
                                        >
                                            Report
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))
                        )}
                    </TableBody>
                </Table>
            </TableContainer>

            {/* Distribution Chart (placeholder) */}
            {stats && stats.by_polymorph_type.length > 0 && (
                <Box sx={{ mt: 4 }}>
                    <Typography variant="h6" gutterBottom>
                        Polymorph Distribution
                    </Typography>
                    <Card>
                        <CardContent>
                            <Grid container spacing={2}>
                                {stats.by_polymorph_type.map((type) => (
                                    <Grid item xs={12} sm={6} md={4} key={type.name}>
                                        <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                                            <Typography variant="subtitle2" color="text.secondary">
                                                {type.name}
                                            </Typography>
                                            <Typography variant="h5">
                                                {type.count}
                                            </Typography>
                                            <Typography variant="caption" color="text.secondary">
                                                {((type.count / stats.total_detections) * 100).toFixed(1)}% of total
                                            </Typography>
                                        </Box>
                                    </Grid>
                                ))}
                            </Grid>
                        </CardContent>
                    </Card>
                </Box>
            )}
        </Container>
    );
}
