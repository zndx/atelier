# ClassificationFlow graph (PR3)

`atelier.flows.classify.ClassificationFlow` is the resident sequence:
start → probe → precondition → load → sweep → fuse → evaluate → publish.

Probe is cheap (no GPU). Expensive precondition only when artifacts are
not signature-final. Evaluate fails unless every target entity has a
`predicted_code`. Sweep LLM is Complete `instruct` (thinking on revisit).
K8s is preferred; the graph is the same as plain `@step`.
