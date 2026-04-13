// Standalone PGlite server for CAI deployment.
// Exposes PostgreSQL wire protocol over TCP so Python (psycopg/SQLAlchemy)
// connects with a standard connection string. pgvector is bundled.
//
// Usage: PGLITE_DATA_DIR=.app/pgdata PGLITE_PORT=5432 node scripts/pglite-server.mjs

import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite/vector'
import { PGLiteSocketServer } from '@electric-sql/pglite-socket'

const DATA_DIR = process.env.PGLITE_DATA_DIR || '.app/pgdata'
const PORT = parseInt(process.env.PGLITE_PORT || '5432', 10)

// Log unhandled errors so crashes leave a trail in the gateway log
// instead of the process silently disappearing and leaving connect()
// attempts to time out.
process.on('unhandledRejection', (reason) => {
  console.error('[pglite] unhandledRejection:', reason)
})
process.on('uncaughtException', (err) => {
  console.error('[pglite] uncaughtException:', err)
  // Crashing loudly is better than hanging — start-app.sh's
  // wait_for_port / supervisor will restart us.
  process.exit(1)
})

const db = await PGlite.create({
  dataDir: DATA_DIR,
  extensions: { vector },
})
console.log(`[pglite] database initialized, data at ${DATA_DIR}`)
await db.exec('CREATE EXTENSION IF NOT EXISTS vector;')
console.log(`[pglite] pgvector extension ready`)

const server = new PGLiteSocketServer({ db, port: PORT, host: '127.0.0.1' })
// Don't await — use .catch() so startup errors crash loudly instead of
// being swallowed by the Promise (upstream issue #964).
server.start()
  .then(() => console.log(`PGlite listening on 127.0.0.1:${PORT}, data at ${DATA_DIR}`))
  .catch(err => {
    console.error('[pglite] FATAL: server.start() failed:', err)
    process.exit(1)
  })

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, async () => {
    try {
      await server.stop()
      await db.close()
    } catch (err) {
      console.error(`[pglite] error during ${sig} shutdown:`, err)
    }
    process.exit(0)
  })
}
