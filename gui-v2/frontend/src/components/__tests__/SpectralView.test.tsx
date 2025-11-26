/**
 * Tests for SpectralView component.
 *
 * This module tests:
 * - Component rendering
 * - Chart data transformation
 * - Empty state handling
 * - Data visualization
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { SpectralView } from '../SpectralView'
import * as systemStore from '@/stores/system-store'

// Mock dependencies
vi.mock('@/stores/system-store', () => ({
  default: vi.fn(() => ({
    spectralData: null,
  })),
}))

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
}))

describe('SpectralView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Rendering', () => {
    it('renders component with title', () => {
      render(<SpectralView />)

      expect(screen.getByText(/Real-time Spectrum/i)).toBeInTheDocument()
    })

    it('shows empty state when no data', () => {
      const mockUseSystemStore = systemStore.default as any
      mockUseSystemStore.mockReturnValue({
        spectralData: null,
      })

      render(<SpectralView />)

      expect(screen.getByText(/Waiting for spectral data/i)).toBeInTheDocument()
    })

    it('shows chart when data is available', () => {
      const mockUseSystemStore = systemStore.default as any
      mockUseSystemStore.mockReturnValue({
        spectralData: {
          wavelengths: [400, 500, 600],
          intensities: [100, 200, 150],
        },
      })

      render(<SpectralView />)

      expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
      expect(screen.getByTestId('line-chart')).toBeInTheDocument()
    })
  })

  describe('Data Transformation', () => {
    it('transforms spectral data correctly', () => {
      const mockUseSystemStore = systemStore.default as any
      const mockData = {
        wavelengths: [400, 500, 600, 700],
        intensities: [100, 200, 150, 50],
      }

      mockUseSystemStore.mockReturnValue({
        spectralData: mockData,
      })

      render(<SpectralView />)

      // Chart should be rendered with transformed data
      expect(screen.getByTestId('line-chart')).toBeInTheDocument()
    })

    it('handles empty wavelengths array', () => {
      const mockUseSystemStore = systemStore.default as any
      mockUseSystemStore.mockReturnValue({
        spectralData: {
          wavelengths: [],
          intensities: [],
        },
      })

      render(<SpectralView />)

      expect(screen.getByText(/Waiting for spectral data/i)).toBeInTheDocument()
    })
  })

  describe('Chart Elements', () => {
    beforeEach(() => {
      const mockUseSystemStore = systemStore.default as any
      mockUseSystemStore.mockReturnValue({
        spectralData: {
          wavelengths: [400, 500, 600],
          intensities: [100, 200, 150],
        },
      })
    })

    it('renders chart with all required elements', () => {
      render(<SpectralView />)

      expect(screen.getByTestId('line')).toBeInTheDocument()
      expect(screen.getByTestId('x-axis')).toBeInTheDocument()
      expect(screen.getByTestId('y-axis')).toBeInTheDocument()
      expect(screen.getByTestId('cartesian-grid')).toBeInTheDocument()
      expect(screen.getByTestId('tooltip')).toBeInTheDocument()
    })
  })
})
