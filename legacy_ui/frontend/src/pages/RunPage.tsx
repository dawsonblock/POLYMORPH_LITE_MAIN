import React from 'react';
import { Container, Box } from '@mui/material';
import ExperimentWizard from '../components/operator/ExperimentWizard';

const RunPage: React.FC = () => {
    return (
        <Container maxWidth="lg">
            <Box sx={{ my: 4 }}>
                <ExperimentWizard />
            </Box>
        </Container>
    );
};

export default RunPage;
