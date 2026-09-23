import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: 'src/tests/audio/workflows/triage/viewer',
  testMatch: '**/*.e2e.spec.mjs',
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: { headless: true, viewport: { width: 1600, height: 1000 } },
})
