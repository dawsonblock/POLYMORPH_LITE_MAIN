import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'sonner'

import { Layout } from '@/components/layout'
import { Dashboard } from '@/pages/dashboard'
import { SystemMonitor } from '@/pages/system-monitor'
import { RecipeManager } from '@/pages/recipe-manager'
import { Analytics } from '@/pages/analytics'
import { Compliance } from '@/pages/compliance'
import { Settings } from '@/pages/settings'
import { LoginPage } from '@/pages/login'

import { useWebSocket } from '@/hooks/use-websocket'
import useAuthStore from '@/stores/auth-store'

function App() {
  const { isAuthenticated } = useAuthStore()
  const { connect, disconnect } = useWebSocket()

  useEffect(() => {
    if (isAuthenticated) {
      connect()
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [isAuthenticated, connect, disconnect])

  if (!isAuthenticated) {
    return (
      <>
        <LoginPage />
        <Toaster position="top-right" richColors />
      </>
    )
  }

  return (
    <>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/system" element={<SystemMonitor />} />
          <Route path="/recipes" element={<RecipeManager />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/compliance" element={<Compliance />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
      <Toaster position="top-right" richColors />
    </>
  )
}

export default App