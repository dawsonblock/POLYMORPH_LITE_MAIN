import { Paper, Typography, Box } from '@mui/material';
import { Science, Inventory2, AccountTree, Assessment } from '@mui/icons-material';

const StatCard = ({
    title,
    value,
    icon,
    color,
}: {
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
}) => (
    <Paper sx={{ p: 3, flex: 1, minWidth: 200 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box
                sx={{
                    bgcolor: color,
                    borderRadius: 2,
                    p: 1.5,
                    mr: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                {icon}
            </Box>
            <Typography variant="h4">{value}</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary">
            {title}
        </Typography>
    </Paper>
);

export default function Dashboard() {
    return (
        <Box>
            <Typography variant="h4" gutterBottom>
                Dashboard
            </Typography>

            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', mt: 2 }}>
                <StatCard
                    title="Total Samples"
                    value={147}
                    icon={<Science />}
                    color="#2196f3"
                />
                <StatCard
                    title="Inventory Items"
                    value={89}
                    icon={<Inventory2 />}
                    color="#4caf50"
                />
                <StatCard
                    title="Active Workflows"
                    value={12}
                    icon={<AccountTree />}
                    color="#ff9800"
                />
                <StatCard
                    title="Completed Today"
                    value={8}
                    icon={<Assessment />}
                    color="#9c27b0"
                />
            </Box>

            <Paper sx={{ p: 3, mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                    Recent Activity
                </Typography>
                <Typography color="text.secondary">
                    No recent activity to display.
                </Typography>
            </Paper>
        </Box>
    );
}
