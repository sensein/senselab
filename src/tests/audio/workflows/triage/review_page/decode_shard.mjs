// Decode one review-page side file with the vendored reader and the page's own expansion, as the page
// does when served, and print its records as JSON: node decode_shard.mjs PAGE_DIR NUMBER [wrapper]
// With "wrapper", read the file:// wrapper's base64 bytes instead of the Parquet file.

import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'

const here = dirname(fileURLToPath(import.meta.url))
const pageScript = join(here, '..', '..', '..', '..', '..', 'senselab', 'audio', 'workflows', 'triage', 'review_page', 'review.js')

const [page, number, mode] = process.argv.slice(2)
vm.runInThisContext(readFileSync(join(page, 'vendor', 'hyparquet.js'), 'utf8'))
const require = createRequire(import.meta.url)
const R = require(pageScript)
const base = join(page, 'data', 'shard-' + String(number).padStart(4, '0'))
let buffer
if (mode === 'wrapper') {
  const text = readFileSync(base + '.js', 'utf8')
  const bytes = Buffer.from(/"([A-Za-z0-9+/=]+)"/.exec(text)[1], 'base64')
  buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
} else {
  const bytes = readFileSync(base + '.parquet')
  buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}
process.stdout.write(JSON.stringify(await R.decodeShard(buffer)))
