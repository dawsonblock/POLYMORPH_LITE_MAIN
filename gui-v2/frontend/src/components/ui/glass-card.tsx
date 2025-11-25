import React from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
    children: React.ReactNode
    className?: string
}

export function GlassCard({ children, className, ...props }: GlassCardProps) {
    return (
        <Card className={cn("glass-card border-none bg-slate-900/40 backdrop-blur-xl", className)} {...props}>
            {children}
        </Card>
    )
}

export { CardHeader, CardFooter, CardTitle, CardDescription, CardContent }
