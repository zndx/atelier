# Vite :3300 — do not advertise Metaflow :3000

Gaius Tilt `kubectl port-forward svc/metaflow-ui-static 3000:3000`
owns :3000. After `systemctl restart atelier`, Vite bound :3001
while Status still advertised :3000.

Pin `ATELIER_VITE_PORT=3300` (`strictPort`). `live_vite_port()` only
counts this checkout's node/vite, never kubectl/tilt.
