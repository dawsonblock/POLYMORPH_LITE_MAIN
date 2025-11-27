import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  FileText,
  Plus,
  Search,
  MoreVertical,
  Play,
  Edit,
  Trash2,
  Clock
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

// Mock recipes
const initialRecipes = [
  { id: 1, name: 'Standard Calibration', type: 'Calibration', duration: '15m', lastRun: '2h ago', status: 'verified' },
  { id: 2, name: 'Hero Crystallization', type: 'Production', duration: '4h 30m', lastRun: '1d ago', status: 'draft' },
  { id: 3, name: 'Quick Scan', type: 'Analysis', duration: '5m', lastRun: '30m ago', status: 'verified' },
  { id: 4, name: 'System Cleaning', type: 'Maintenance', duration: '45m', lastRun: '1w ago', status: 'archived' },
]

export function RecipeManager() {
  const [recipes] = useState(initialRecipes)
  const [searchQuery, setSearchQuery] = useState('')

  const filteredRecipes = recipes.filter(r =>
    r.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'verified': return <Badge variant="success">Verified</Badge>
      case 'draft': return <Badge variant="warning">Draft</Badge>
      case 'archived': return <Badge variant="secondary">Archived</Badge>
      default: return <Badge variant="outline">{status}</Badge>
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Recipe Manager</h1>
          <p className="text-muted-foreground">Create and manage experimental workflows</p>
        </div>
        <Button variant="scientific">
          <Plus className="mr-2 h-4 w-4" />
          New Recipe
        </Button>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search recipes..."
            className="pl-9"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        {/* Filter buttons could go here */}
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {filteredRecipes.map((recipe) => (
          <motion.div
            key={recipe.id}
            layout
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <Card className="hover:border-primary/50 transition-colors cursor-pointer group">
              <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
                <div className="space-y-1">
                  <CardTitle className="text-base font-semibold group-hover:text-primary transition-colors">
                    {recipe.name}
                  </CardTitle>
                  <CardDescription>{recipe.type}</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusBadge(recipe.status)}
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-4 text-sm text-muted-foreground mt-2">
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {recipe.duration}
                  </div>
                  <div className="flex items-center gap-1">
                    <FileText className="h-3 w-3" />
                    Last run: {recipe.lastRun}
                  </div>
                </div>

                <div className="flex items-center gap-2 mt-4 pt-4 border-t opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button size="sm" variant="default" className="w-full">
                    <Play className="mr-2 h-3 w-3" />
                    Run
                  </Button>
                  <Button size="sm" variant="outline" className="w-full">
                    <Edit className="mr-2 h-3 w-3" />
                    Edit
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}