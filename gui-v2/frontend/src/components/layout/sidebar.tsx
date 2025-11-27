import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
    LayoutDashboard,
    Activity,
    FlaskConical,
    BarChart3,
    ShieldCheck,
    Settings,
    LogOut,
    Beaker
} from 'lucide-react'
import useAuthStore from '@/stores/auth-store'

const navItems = [
    { icon: LayoutDashboard, label: 'Dashboard', href: '/dashboard' },
    { icon: Activity, label: 'System Monitor', href: '/system' },
    { icon: FlaskConical, label: 'Recipes', href: '/recipes' },
    { icon: BarChart3, label: 'Analytics', href: '/analytics' },
    { icon: ShieldCheck, label: 'Compliance', href: '/compliance' },
    { icon: Settings, label: 'Settings', href: '/settings' },
]

export function Sidebar() {
    const location = useLocation()
    const { logout } = useAuthStore()

    return (
        <div className="flex h-screen w-64 flex-col border-r bg-card/30 backdrop-blur-xl">
            <div className="flex h-16 items-center border-b px-6">
                <Beaker className="mr-2 h-6 w-6 text-primary animate-pulse" />
                <span className="text-lg font-bold tracking-tight text-foreground">
                    POLYMORPH<span className="text-primary">LITE</span>
                </span>
            </div>

            <div className="flex-1 overflow-y-auto py-4">
                <nav className="grid gap-1 px-2">
                    {navItems.map((item, index) => {
                        const isActive = location.pathname === item.href
                        return (
                            <Link
                                key={index}
                                to={item.href}
                                className={cn(
                                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all hover:text-primary",
                                    isActive
                                        ? "bg-primary/10 text-primary shadow-sm border border-primary/20"
                                        : "text-muted-foreground hover:bg-accent"
                                )}
                            >
                                <item.icon className="h-4 w-4" />
                                {item.label}
                            </Link>
                        )
                    })}
                </nav>
            </div>

            <div className="border-t p-4">
                <button
                    onClick={logout}
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-muted-foreground transition-all hover:bg-destructive/10 hover:text-destructive"
                >
                    <LogOut className="h-4 w-4" />
                    Sign Out
                </button>
            </div>
        </div>
    )
}
