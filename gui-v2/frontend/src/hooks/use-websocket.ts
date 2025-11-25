import { useCallback, useEffect, useRef } from 'react'
import { io, Socket } from 'socket.io-client'
import { toast } from 'sonner'

import useSystemStore from '@/stores/system-store'
import useAuthStore from '@/stores/auth-store'
import type { SystemStatus, Process, Alert, SpectralData } from '@/types'

// Simple throttle implementation
function throttle<T extends (...args: any[]) => any>(func: T, limit: number): T {
  let inThrottle: boolean
  return function (this: any, ...args: any[]) {
    if (!inThrottle) {
      func.apply(this, args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  } as T
}

export const useWebSocket = () => {
  const socketRef = useRef<Socket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

  const { token } = useAuthStore()
  const {
    setStatus,
    setProcesses,
    addAlert,
    setConnectionStatus,
    setSpectralData,
  } = useSystemStore()

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return
    }

    socketRef.current = io('http://localhost:8001', {
      auth: {
        token: token,
      },
      transports: ['websocket'],
    })

    const socket = socketRef.current

    const handleSystemStatus = (data: SystemStatus) => setStatus(data)
    const handleProcessesUpdate = (data: Process[]) => setProcesses(data)
    const handleNewAlert = (alert: Alert) => {
      addAlert(alert)
      const toastFn = {
        info: toast.info,
        warning: toast.warning,
        error: toast.error,
        success: toast.success,
      }[alert.type]

      toastFn(alert.message, {
        description: `From ${alert.source}`,
        action: {
          label: 'Acknowledge',
          onClick: () => {
            emit('acknowledge_alert', { alertId: alert.id });
          },
        },
      })
    }
    // Throttle spectral data to ~30fps (33ms)
    const handleSpectralData = throttle((data: SpectralData) => {
      setSpectralData(data)
    }, 33)

    socket.on('connect', () => {
      console.log('✅ WebSocket connected')
      setConnectionStatus(true)
      toast.success('Connected to system')
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    })

    socket.on('disconnect', (reason) => {
      console.log('❌ WebSocket disconnected:', reason)
      setConnectionStatus(false)
      toast.error('Disconnected from system')
      reconnectTimeoutRef.current = setTimeout(() => {
        if (!socket.connected) {
          console.log('🔄 Attempting to reconnect...')
          socket.connect()
        }
      }, 5000)
    })

    socket.on('system_status', handleSystemStatus)
    socket.on('processes_update', handleProcessesUpdate)
    socket.on('new_alert', handleNewAlert)
    socket.on('spectral_data', handleSpectralData)

    socket.on('connect_error', (error) => {
      console.error('❌ WebSocket connection error:', error)
      setConnectionStatus(false)
      toast.error('Connection failed')
    })

    // Cleanup function for this connection attempt
    return () => {
      socket.off('system_status', handleSystemStatus)
      socket.off('processes_update', handleProcessesUpdate)
      socket.off('new_alert', handleNewAlert)
      socket.off('spectral_data', handleSpectralData)
      socket.disconnect()
    }

  }, [token, setStatus, setProcesses, addAlert, setConnectionStatus, setSpectralData])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (socketRef.current) {
      socketRef.current.disconnect()
      socketRef.current = null
    }
    setConnectionStatus(false)
  }, [setConnectionStatus])

  const emit = useCallback((event: string, data?: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data)
    }
  }, [])

  // Auto-connect on mount
  useEffect(() => {
    const cleanup = connect()
    return () => {
      if (cleanup) cleanup()
      disconnect()
    }
  }, [connect, disconnect])

  return {
    connect,
    disconnect,
    emit,
    isConnected: socketRef.current?.connected || false,
  }
}