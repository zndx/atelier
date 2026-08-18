# Waffle self-link opened about:blank

Copy-link showed `http://192.168.1.55:3300/`; click opened a new tab at
`about:blank`. That is Chrome on `target="_blank" rel="noopener"` when
the href is the current document (home `/`). Peer ports were fine.

Left-click on the current document now `window.open` + `location.replace`
so the new tab actually navigates. Rail stays mounted (`hidden`) like
Signals.
