# Atelier waffle / ServerQuery adoption

Pinned `external/signals-protocol` to `25c54ab` (Status.surfaces +
ServerQuery). Regenerated `src/zndx/engine/v1` stubs.

- Status advertises `kind=primary` on devenv Vite `:3000` (never loopback)
- ServerQuery REMOTES = `git remote -v` + HEAD; PEERS share Signals host
- Gateway `GET /api/atelier/v1/federation/surfaces`
- Chrome 9-dot waffle; LAN IP open rebases links to `ip:<port>`

Gaius/Signals/Ægir appear on the same advertise host. Engine process
must restart to serve ServerQuery on `:50251`.
