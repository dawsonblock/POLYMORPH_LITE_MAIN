import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type { SystemStatus, Process, Alert, SpectralData } from '@/types'

interface SystemState {
  status: SystemStatus | null
  activeProcesses: Process[]
  alerts: Alert[]
  spectralData: SpectralData | null
  isConnected: boolean
  lastUpdate: Date | null

  // Actions
  setStatus: (status: SystemStatus) => void
  setProcesses: (processes: Process[]) => void
  addAlert: (alert: Alert) => void
  setSpectralData: (data: SpectralData) => void
  acknowledgeAlert: (alertId: string) => void
  setConnectionStatus: (connected: boolean) => void
  updateLastUpdate: () => void
}

const useSystemStore = create<SystemState>()(
  devtools(
    (set) => ({
      status: null,
      activeProcesses: [],
      alerts: [],
      spectralData: null,
      isConnected: false,
      lastUpdate: null,

      setStatus: (status) =>
        set(
          { status, lastUpdate: new Date() },
          false,
          'system/setStatus'
        ),

      setProcesses: (activeProcesses) =>
        set(
          { activeProcesses, lastUpdate: new Date() },
          false,
          'system/setProcesses'
        ),

      addAlert: (alert) =>
        set(
          (state) => ({
            alerts: [alert, ...state.alerts].slice(0, 100), // Keep only last 100 alerts
            lastUpdate: new Date(),
          }),
          false,
          'system/addAlert'
        ),

      setSpectralData: (data) =>
        set(
          { spectralData: data },
          false,
          'system/setSpectralData'
        ),

      acknowledgeAlert: (alertId) =>
        set(
          (state) => ({
            alerts: state.alerts.map((alert) =>
              alert.id === alertId ? { ...alert, acknowledged: true } : alert
            ),
          }),
          false,
          'system/acknowledgeAlert'
        ),

      setConnectionStatus: (isConnected) =>
        set(
          { isConnected, lastUpdate: new Date() },
          false,
          'system/setConnectionStatus'
        ),

      updateLastUpdate: () =>
        set(
          { lastUpdate: new Date() },
          false,
          'system/updateLastUpdate'
        ),
    }),
    {
      name: 'system-store',
    }
  )
)

export default useSystemStore