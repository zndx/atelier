# RequestQueueShare on WRK admit

Mirror Gaius `queue_share.py`. `ensure()` (vLLM launch) calls
`RequestQueueShare` with occupancy intent for instruct/referee on
`root.internal.inference.instruct`. `QueueHint` remains declared leaf
shape (`gpu_guarantee=0`). UNIMPLEMENTED = Signals-not-yet; other
errors `#YK.00000007.SHAREFAIL`. Never write queues.yaml.
