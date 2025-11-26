import { test, expect } from '@playwright/test'

test.describe('Login Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should display login form', async ({ page }) => {
    await expect(page.getByRole('heading', { name: /POLYMORPH-4 Lite/i })).toBeVisible()
    await expect(page.getByLabel(/username/i)).toBeVisible()
    await expect(page.getByLabel(/password/i)).toBeVisible()
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible()
  })

  test('should show validation error for empty credentials', async ({ page }) => {
    await page.getByRole('button', { name: /sign in/i }).click()

    // Toast notification should appear
    await expect(page.getByText(/please enter both username and password/i)).toBeVisible()
  })

  test('should toggle password visibility', async ({ page }) => {
    const passwordInput = page.getByLabel(/password/i)
    const toggleButton = page.getByRole('button', { name: '' }).first() // Eye icon button

    // Password should be hidden initially
    await expect(passwordInput).toHaveAttribute('type', 'password')

    // Click to show password
    await toggleButton.click()
    await expect(passwordInput).toHaveAttribute('type', 'text')

    // Click to hide password again
    await toggleButton.click()
    await expect(passwordInput).toHaveAttribute('type', 'password')
  })

  test('should successfully login with valid credentials', async ({ page }) => {
    // Fill in login form
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')

    // Submit form
    await page.getByRole('button', { name: /sign in/i }).click()

    // Should redirect to dashboard
    await expect(page).toHaveURL(/dashboard/)

    // Should show success message
    await expect(page.getByText(/welcome back/i)).toBeVisible()
  })

  test('should show error for invalid credentials', async ({ page }) => {
    await page.getByLabel(/username/i).fill('wronguser')
    await page.getByLabel(/password/i).fill('wrongpass')

    await page.getByRole('button', { name: /sign in/i }).click()

    // Should show error toast
    await expect(page.getByText(/login failed/i)).toBeVisible()

    // Should stay on login page
    await expect(page).toHaveURL(/\/$/)
  })

  test('should disable form during login', async ({ page }) => {
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')

    // Click sign in
    const signInButton = page.getByRole('button', { name: /sign in/i })
    await signInButton.click()

    // Button should show loading state
    await expect(page.getByText(/signing in/i)).toBeVisible()

    // Inputs should be disabled
    await expect(page.getByLabel(/username/i)).toBeDisabled()
    await expect(page.getByLabel(/password/i)).toBeDisabled()
  })
})

test.describe('Session Management', () => {
  test('should persist login across page reloads', async ({ page }) => {
    // Login first
    await page.goto('/')
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')
    await page.getByRole('button', { name: /sign in/i }).click()

    await expect(page).toHaveURL(/dashboard/)

    // Reload page
    await page.reload()

    // Should still be logged in (not redirected to login)
    await expect(page).toHaveURL(/dashboard/)
  })

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.goto('/')
    await page.getByLabel(/username/i).fill('admin')
    await page.getByLabel(/password/i).fill('admin')
    await page.getByRole('button', { name: /sign in/i }).click()

    await expect(page).toHaveURL(/dashboard/)

    // Find and click logout button
    await page.getByRole('button', { name: /logout/i }).click()

    // Should redirect to login page
    await expect(page).toHaveURL(/\/$/)
  })
})
