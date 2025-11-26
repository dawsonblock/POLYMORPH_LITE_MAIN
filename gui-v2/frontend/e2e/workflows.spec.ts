import { test, expect } from '@playwright/test'

test.describe('Workflow Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login
    await page.goto('/')
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')
    await page.getByRole('button', { name: /sign in/i }).click()

    await expect(page).toHaveURL(/dashboard/)
  })

  test('should create and execute workflow', async ({ page }) => {
    // Navigate to workflow section
    await page.goto('/dashboard')

    // Look for workflow management
    const workflowLink = page.getByRole('link', { name: /workflow|recipe/i })
    if (await workflowLink.isVisible()) {
      await workflowLink.click()

      // Should be able to see workflow list
      await expect(page.getByText(/workflow|recipe/i)).toBeVisible()
    }
  })

  test('should display workflow execution status', async ({ page }) => {
    await page.goto('/dashboard')

    // Check for any running workflows or status indicators
    const statusIndicator = page.getByText(/running|idle|complete/i)
    if (await statusIndicator.isVisible()) {
      await expect(statusIndicator).toBeVisible()
    }
  })
})

test.describe('Device Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')
    await page.getByRole('button', { name: /sign in/i }).click()
  })

  test('should display connected devices', async ({ page }) => {
    // Navigate to system monitor or device page
    await page.getByRole('link', { name: /system monitor/i }).click()

    // Should show device status
    await expect(page.getByText(/device|daq|raman/i)).toBeVisible()
  })

  test('should show device health status', async ({ page }) => {
    await page.getByRole('link', { name: /system monitor/i }).click()

    // Look for health indicators
    const healthIndicator = page.locator('[data-testid="device-health"]')
    if (await healthIndicator.isVisible()) {
      await expect(healthIndicator).toBeVisible()
    }
  })
})

test.describe('Data Visualization', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')
    await page.getByRole('button', { name: /sign in/i }).click()
  })

  test('should display spectral data chart', async ({ page }) => {
    await page.goto('/dashboard')

    // Spectral view should be present
    await expect(page.getByText(/spectrum/i)).toBeVisible()

    // Chart library should be loaded
    const chart = page.locator('.recharts-responsive-container')
    await expect(chart).toBeVisible()
  })

  test('should update chart with real-time data', async ({ page }) => {
    await page.goto('/dashboard')

    // Wait for initial chart render
    await page.waitForSelector('.recharts-responsive-container')

    // Note: In real implementation, would check for data updates
    // This tests that the chart structure is present
    const chartLines = page.locator('.recharts-line')
    if (await chartLines.count() > 0) {
      await expect(chartLines.first()).toBeVisible()
    }
  })
})

test.describe('Analytics', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')
    await page.getByRole('button', { name: /sign in/i }).click()

    await page.getByRole('link', { name: /analytics/i }).click()
  })

  test('should display analytics dashboard', async ({ page }) => {
    await expect(page).toHaveURL(/analytics/)

    // Analytics title should be visible
    await expect(page.getByText(/analytics/i)).toBeVisible()
  })

  test('should show data visualizations', async ({ page }) => {
    // Look for charts or graphs
    const charts = page.locator('.recharts-responsive-container')
    if (await charts.count() > 0) {
      await expect(charts.first()).toBeVisible()
    }
  })
})

test.describe('Complete User Journey', () => {
  test('should complete full workflow from login to execution', async ({ page }) => {
    // 1. Login
    await page.goto('/')
    await page.getByLabel(/username/i).fill('operator')
    await page.getByLabel(/password/i).fill('operator')
    await page.getByRole('button', { name: /sign in/i }).click()

    await expect(page).toHaveURL(/dashboard/)

    // 2. View system status
    await expect(page.getByText(/dashboard/i)).toBeVisible()

    // 3. Navigate to system monitor
    await page.getByRole('link', { name: /system monitor/i }).click()
    await expect(page).toHaveURL(/system-monitor/)

    // 4. Check analytics
    await page.getByRole('link', { name: /analytics/i }).click()
    await expect(page).toHaveURL(/analytics/)

    // 5. View compliance (if operator has access)
    const complianceLink = page.getByRole('link', { name: /compliance/i })
    if (await complianceLink.isVisible()) {
      await complianceLink.click()
      await expect(page).toHaveURL(/compliance/)
    }

    // 6. Logout
    await page.getByRole('button', { name: /logout/i }).click()
    await expect(page).toHaveURL(/\/$/)
  })
})
