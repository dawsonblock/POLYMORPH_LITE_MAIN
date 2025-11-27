import { ReactNode } from 'react'
import { Sidebar } from './sidebar'
import { Header } from './header'

interface LayoutProps {
    children: ReactNode
}

export function Layout({ children }: LayoutProps) {
    return (
        <div className="flex h-screen overflow-hidden bg-background">
            <Sidebar />
            <div className="flex flex-1 flex-col overflow-hidden">
                <Header />
                <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
                    <div className="mx-auto max-w-7xl animate-fade-in">
                        {children}
                    </div>
                </main>
            </div>
        </div>
    )
}
