import { Bell, User, Wifi, WifiOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useWebSocket } from '@/hooks/use-websocket'
import useAuthStore from '@/stores/auth-store'
import { Badge } from '@/components/ui/badge'

export function Header() {
    const { isConnected } = useWebSocket()
    const { user } = useAuthStore()

    return (
        <header className="flex h-16 items-center justify-between border-b bg-card/30 px-6 backdrop-blur-xl">
            <div className="flex items-center gap-4">
                {/* Breadcrumbs or Page Title could go here */}
                <Badge variant={isConnected ? "success" : "destructive"} pulsing={isConnected} className="gap-1">
                    {isConnected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
                    {isConnected ? "System Online" : "Disconnected"}
                </Badge>
            </div>

            <div className="flex items-center gap-4">
                <Button variant="ghost" size="icon" className="relative">
                    <Bell className="h-5 w-5 text-muted-foreground" />
                    <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-primary" />
                </Button>

                <div className="flex items-center gap-3 border-l pl-4">
                    <div className="flex flex-col items-end">
                        <span className="text-sm font-medium leading-none">{user?.username || 'User'}</span>
                        <span className="text-xs text-muted-foreground">{user?.role || 'Scientist'}</span>
                    </div>
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 border border-primary/20">
                        <User className="h-4 w-4 text-primary" />
                    </div>
                </div>
            </div>
        </header>
    )
}
