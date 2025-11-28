import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Box,
    Paper,
    Typography,
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    TextField,
    InputAdornment,
} from '@mui/material';
import { Add, Search } from '@mui/icons-material';
import { samplesApi } from '../../api/samples';

export default function SampleList() {
    const [searchTerm, setSearchTerm] = useState('');

    const { data: samples, isLoading } = useQuery({
        queryKey: ['samples'],
        queryFn: samplesApi.list,
    });

    const filteredSamples = samples?.filter((sample) =>
        sample.sample_id.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const getStatusColor = (status: string) => {
        switch (status.toLowerCase()) {
            case 'active':
                return 'success';
            case 'pending':
                return 'warning';
            case 'completed':
                return 'info';
            default:
                return 'default';
        }
    };

    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4">Samples</Typography>
                <Button variant="contained" startIcon={<Add />}>
                    New Sample
                </Button>
            </Box>

            <Paper sx={{ p: 2, mb: 2 }}>
                <TextField
                    placeholder="Search samples..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    fullWidth
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <Search />
                            </InputAdornment>
                        ),
                    }}
                />
            </Paper>

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Sample ID</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Created At</TableCell>
                            <TableCell>Parent ID</TableCell>
                            <TableCell>Actions</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {isLoading && (
                            <TableRow>
                                <TableCell colSpan={5} align="center">
                                    Loading...
                                </TableCell>
                            </TableRow>
                        )}
                        {!isLoading && filteredSamples?.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={5} align="center">
                                    No samples found
                                </TableCell>
                            </TableRow>
                        )}
                        {filteredSamples?.map((sample) => (
                            <TableRow key={sample.id} hover>
                                <TableCell>{sample.sample_id}</TableCell>
                                <TableCell>
                                    <Chip
                                        label={sample.status}
                                        color={getStatusColor(sample.status)}
                                        size="small"
                                    />
                                </TableCell>
                                <TableCell>
                                    {new Date(sample.created_at).toLocaleDateString()}
                                </TableCell>
                                <TableCell>{sample.parent_id || '-'}</TableCell>
                                <TableCell>
                                    <Button size="small">View</Button>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
}
