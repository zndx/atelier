# One vLLM process, the deployed model selects the mix

2026-10-11. Follow-up to `212437_decisions-queue-excludes-heavy.md`.

Gaius pins vLLM 0.27.1. The upgrade of that binary is what lets the
one process serve `thinking`, `instruct`, and `decisions`. Which of
those it actually serves is the model Airflow has deployed.

The process is the existing tensor-parallel-4 server. One port. With
`Qwen/Qwen3.8-27B` loaded it advertises `thinking` and `instruct` on
`/v1/chat/completions`. With `vllm-sr/d3` loaded it advertises
`decisions` on `POST /v1/decisions`. The route exists on that process
in both deployments. The loaded weights are what make a capability
healthy. A request for `decisions` while Qwen is loaded fails closed,
and a request for `thinking` while d3 is loaded fails closed.

Airflow is what changes the weights. The user activity names the
workload, the workload names the capability mix, and the mix names the
model. Admitting classification deploys `vllm-sr/d3` into this process.
Releasing it deploys `Qwen/Qwen3.8-27B` again, before the revisit.
`start_model` already takes a `model_id` for an endpoint. The activity
uses that on the one process. It does not start a second server.

The queues stay mutually exclusive on this host. `heavy` is the
workload whose model is Qwen. `decisions` is the workload whose model
is d3. Both want the same four cards, and parent max is 6, so the
arbiter holds only one floor at 4. The queue transition and the model
deployment are the same Airflow step: retract heavy, assert
`root.internal.inference.decisions` at 4, deploy d3, wait until
`decisions` is serving, run the `@kubernetes` sweep. `all_done`
reverses it: deploy Qwen, restore the heavy floor, wait until
`thinking` is serving, run the revisit.
