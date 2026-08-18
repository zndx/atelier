# Restored app shell after boot regression

Ghostty preload in `main.tsx` / `index.html` plus the lifted Layout
around Routes left a blank page. Restored `App.tsx`, `main.tsx`,
`index.html`, and `Layout.tsx` from trunk. Waffle links are Ægir-style
plain `<a target="_blank">` (no click interceptor); rail stays portaled
so Ant header overflow does not clip it.
