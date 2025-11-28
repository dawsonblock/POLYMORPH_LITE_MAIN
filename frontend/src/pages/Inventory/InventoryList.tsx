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
    Alert,
} from '@mui/material';
import { Add, Warning } from '@mui/icons-material';
import { inventoryApi } from '../../api/inventory';

export default function InventoryList() {
    const { data: items, isLoading } = useQuery({
        queryKey: ['inventory'],
        queryFn: inventoryApi.list,
    });

    const { data: lowStockItems } = useQuery({
        queryKey: ['inventory', 'low-stock'],
        queryFn: inventoryApi.getLowStock,
    });

    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h4">Inventory</Typography>
                <Button variant="contained" startIcon={<Add />}>
                    New Item
                </Button>
            </Box>

            {lowStockItems && lowStockItems.length > 0 && (
                <Alert severity="warning" icon={<Warning />} sx={{ mb: 2 }}>
                    {lowStockItems.length} item(s) are running low on stock
                </Alert>
            )}

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Item ID</TableCell>
                            <TableCell>Name</TableCell>
                            <TableCell>Category</TableCell>
                            <TableCell>Stock</TableCell>
                            <TableCell>Unit</TableCell>
                            <TableCell>Min Stock</TableCell>
                            <TableCell>Status</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {isLoading && (
                            <TableRow>
                                <TableCell colSpan={7} align="center">
                                    Loading...
                                </TableCell>
                            </TableRow>
                        )}
                        {!isLoading && items?.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={7} align="center">
                                    No inventory items found
                                </TableCell>
                            </TableRow>
                        )}
                        {items?.map((item) => {
                            const isLowStock = item.stock_quantity <= item.min_stock_level;
                            return (
                                <TableRow key={item.id} hover sx={isLowStock ? { bgcolor: 'warning.dark' } : {}}>
                                    <TableCell>{item.item_id}</TableCell>
                                    <TableCell>{item.name}</TableCell>
                                    <TableCell>{item.category}</TableCell>
                                    <TableCell>{item.stock_quantity}</TableCell>
                                    <TableCell>{item.unit}</TableCell>
                                    <TableCell>{item.min_stock_level}</TableCell>
                                    <TableCell>
                                        {isLowStock ? (
                                            <Typography color="warning.main" variant="body2">
                                                Low Stock
                                            </Typography>
                                        ) : (
                                            <Typography color="success.main" variant="body2">
                                                In Stock
                                            </Typography>
                                        )}
                                    </TableCell>
                                </TableRow>
                            );
                        })}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
}
