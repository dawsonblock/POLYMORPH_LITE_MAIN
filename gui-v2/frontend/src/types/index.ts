export interface User {
  id: string
  username: string
  email: string
  role: 'admin' | 'operator' | 'viewer'
  lastLogin: Date
  isActive: boolean
}

export interface SystemStatus {
  overall: 'healthy' | 'warning' | 'error' | 'offline'
  components: {
    daq: ComponentStatus
    raman: ComponentStatus
    safety: ComponentStatus
    pumps: ComponentStatus
    valves: ComponentStatus
  }
  uptime: number
  lastUpdate: Date
}

export interface ComponentStatus {
  status: 'online' | 'offline' | 'error' | 'warning'
  temperature?: number
  pressure?: number
  flowRate?: number
  errorCode?: string
  lastMaintenance?: Date
}

export interface Recipe {
  id: string
  name: string
  description: string
  version: string
  author: string
  createdAt: Date
  updatedAt: Date
  steps: RecipeStep[]
  parameters: Record<string, any>
  isActive: boolean
  tags: string[]
}

export interface RecipeStep {
  id: string
  name: string
  type: 'pump' | 'valve' | 'wait' | 'sample' | 'clean'
  duration: number
  parameters: Record<string, any>
  conditions?: Record<string, any>
}

export interface Process {
  id: string
  recipeId: string
  recipeName: string
  status: 'running' | 'completed' | 'failed' | 'paused'
  startTime: Date
  endTime?: Date
  currentStep: number
  totalSteps: number
  progress: number
  data: ProcessData[]
  alerts: Alert[]
}

export interface ProcessData {
  timestamp: Date
  step: number
  temperature: number
  pressure: number
  flowRate: number
  ph?: number
  concentration?: number
}

export interface Alert {
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  message: string
  timestamp: Date
  acknowledged: boolean
  source: string
}

export interface AnalyticsData {
  processEfficiency: number
  averageRuntime: number
  errorRate: number
  maintenanceAlerts: number
  utilizationRate: number
  qualityMetrics: {
    yield: number
    purity: number
    consistency: number
  }
}

export interface ComplianceRecord {
  id: string
  type: 'audit_trail' | 'electronic_signature' | 'batch_record'
  timestamp: Date
  userId: string
  action: string
  details: Record<string, any>
  signature?: string
}

export interface SpectralData {
  t: number
  wavelengths: number[]
  intensities: number[]
  peak_nm: number
  peak_intensity: number
}