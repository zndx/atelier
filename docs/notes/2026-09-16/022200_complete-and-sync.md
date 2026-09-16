# SyncWorkloads + Engine/Complete (critical phase)

WorkloadSync replace-submits paused `atelier_sdg_classify` to Signals.
Classification LLM is `Engine/Complete` (`instruct` first sweep, `thinking`
on revisit). No Anthropic/Bedrock default. Probe/precondition steps call
resident helpers; expensive precondition still only when SKOS/encoder stale.
