import { motion } from 'framer-motion'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function Compliance() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <Card>
        <CardHeader>
          <CardTitle>Compliance Panel</CardTitle>
        </CardHeader>
        <CardContent>
          <p>21 CFR Part 11 compliance features coming soon...</p>
        </CardContent>
      </Card>
    </motion.div>
  )
}