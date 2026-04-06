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

const db = await PGlite.create({
  dataDir: DATA_DIR,
  extensions: { vector },
})
await db.exec('CREATE EXTENSION IF NOT EXISTS vector;')

const server = new PGLiteSocketServer({ db, port: PORT, host: '127.0.0.1' })
await server.start()
console.log(`PGlite listening on 127.0.0.1:${PORT}, data at ${DATA_DIR}`)

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, async () => {
    await server.stop()
    await db.close()
    process.exit(0)
  })
}
