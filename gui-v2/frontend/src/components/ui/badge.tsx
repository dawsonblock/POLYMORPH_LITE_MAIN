import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline: "text-foreground",
        success: "border-transparent bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/25 border-emerald-500/20",
        warning: "border-transparent bg-amber-500/15 text-amber-400 hover:bg-amber-500/25 border-amber-500/20",
        error: "border-transparent bg-rose-500/15 text-rose-400 hover:bg-rose-500/25 border-rose-500/20",
        info: "border-transparent bg-sky-500/15 text-sky-400 hover:bg-sky-500/25 border-sky-500/20",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
  VariantProps<typeof badgeVariants> {
  pulsing?: boolean
}

function Badge({ className, variant, pulsing, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props}>
      {pulsing && (
        <span className="mr-1.5 flex h-2 w-2 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
        </span>
      )}
      {props.children}
    </div>
  )
}

export { Badge, badgeVariants }