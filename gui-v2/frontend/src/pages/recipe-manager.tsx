import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function RecipeManager() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <Card>
        <CardHeader>
          <CardTitle>Recipe Manager</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Recipe management features coming soon...</p>
        </CardContent>
      </Card>
    </motion.div>
  )
}