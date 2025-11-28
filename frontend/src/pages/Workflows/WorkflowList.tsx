import { useQuery } from '@tanstack/react-query';
import {
    Box,
    Paper,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
} from '@mui/material';
import { workflowsApi } from '../../api/workflows';

export default function WorkflowList() {
    const { data: workflows, isLoading } = useQuery({
        queryKey: ['workflows'],
        queryFn: workflowsApi.list,
    });

    const getStatusColor = (status: string) => {
        switch (status.toLowerCase()) {
            case 'running':
                return 'primary';
            case 'completed':
                return 'success';
            case 'failed':
                return 'error';
            case 'aborted':
                return 'warning';
            default:
                return 'default';
        }
    };

    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Workflow Executions
            </Typography>

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Run ID</TableCell>
                            <TableCell>Workflow Name</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Operator</TableCell>
                            <TableCell>Started At</TableCell>
                            <TableCell>Completed At</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {isLoading && (
                            <TableRow>
                                <TableCell colSpan={6} align="center">
                                    Loading...
                                </TableCell>
                            </TableRow>
                        )}
                        {!isLoading && workflows?.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={6} align="center">
                                    No workflow executions found
                                </TableCell>
                            </TableRow>
                        )}
                        {workflows?.map((workflow) => (
                            <TableRow key={workflow.id} hover>
                                <TableCell>{workflow.run_id}</TableCell>
                                <TableCell>{workflow.workflow_name}</TableCell>
                                <TableCell>
                                    <Chip
                                        label={workflow.status}
                                        color={getStatusColor(workflow.status)}
                                        size="small"
                                    />
                                </TableCell>
                                <TableCell>{workflow.operator}</TableCell>
                                <TableCell>
                                    {new Date(workflow.started_at).toLocaleString()}
                                </TableCell>
                                <TableCell>
                                    {workflow.completed_at
                                        ? new Date(workflow.completed_at).toLocaleString()
                                        : '-'}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
}
