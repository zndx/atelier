# Queue share: UUIDv7, heavy leaf, end window

Every `RequestQueueShare` mints RFC 9562 UUIDv7 (no omit, no v4). Admit
sends occupancy; WRK stop sends zero-floor + `valid_until_ns` and
supersedes the admit id. `QUEUE_SHARE_REJECTED` is SHAREFAIL — do not
admit. TP=4 instruct/referee occupy `root.internal.inference.heavy`
(same leaf as Gaius/Ægir). Never write queues.yaml.
