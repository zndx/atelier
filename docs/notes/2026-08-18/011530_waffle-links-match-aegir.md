# Waffle links: match Ægir/Signals, drop window.open

Custom left-click (`preventDefault` + `window.open`) broke every
federation link. Ægir, Signals, and Gaius use a plain
`<a href target="_blank" rel="noopener noreferrer">` with no handler —
including the same-port self-link. Atelier waffle now matches that.
Rail stays in-tree (`hidden`); `.atelier-chrome` is `overflow: visible`
so Ant Header does not clip it.
