"""Step definitions for GPU acceleration config + preflight scenarios."""

from __future__ import annotations

from behave import given, when, then


# ── preflight ──────────────────────────────────────────────────────


@when("I call preflight_gpu")
def step_preflight(context):
    from atelier.classify.gpu import preflight_gpu
    context.gpu = preflight_gpu()


@then("the result has a resolved_devices list")
def step_resolved_list(context):
    assert isinstance(context.gpu.resolved_devices, list), (
        f"resolved_devices type: {type(context.gpu.resolved_devices)}"
    )
    assert len(context.gpu.resolved_devices) >= 1


@then("resolved_device matches the first entry's prefix")
def step_resolved_prefix(context):
    first = context.gpu.resolved_devices[0]
    assert first.startswith(context.gpu.resolved_device), (
        f"{first!r} does not start with {context.gpu.resolved_device!r}"
    )


# ── Config fields ──────────────────────────────────────────────────


@when("I load the atelier config")
def step_load_config(context):
    from atelier.config import load_config
    context.cfg = load_config()


@then("the config has {attr}")
def step_has_attr(context, attr):
    assert hasattr(context.cfg, attr), f"Missing attribute: {attr}"


# ── Acceleration endpoint ──────────────────────────────────────────


@when("I call GET /api/acceleration")
def step_get_acceleration(context):
    from fastapi.testclient import TestClient
    from atelier.gateway import app
    c = TestClient(app)
    r = c.get("/api/acceleration")
    assert r.status_code == 200, r.text
    context.accel = r.json()


@then("the response has a methods object")
def step_methods(context):
    assert "methods" in context.accel
    assert isinstance(context.accel["methods"], dict)
    for key in ("sage_gpu", "shap_gpu", "catboost_gpu", "embedding_sharded"):
        assert key in context.accel["methods"], f"missing methods.{key}"


@then("the response has a config object")
def step_config(context):
    assert "config" in context.accel
    for key in ("gpu_enabled", "shard_threshold", "sage_chunk"):
        assert key in context.accel["config"], f"missing config.{key}"


@then("the response summary is a non-empty string")
def step_summary(context):
    s = context.accel.get("summary", "")
    assert isinstance(s, str) and len(s) > 0


# ── Encoder lazy load ──────────────────────────────────────────────


@given("embedding configured with at least one device")
def step_embed_configure(context):
    from atelier.classify.embedding import configure
    # Force fresh state — tests shouldn't rely on whatever prior runs set.
    configure(device="auto", batch_size=32, shard_threshold=200_000)


@when("I fetch the encoder")
def step_fetch_encoder(context):
    from atelier.classify.embedding import _get_model
    context.encoder = _get_model()


@then("only the first replica is loaded initially")
def step_one_replica(context):
    # Internal inspection: _models[0] non-None, rest may be None until sharding
    # engages.  This proves the lazy init path in MultiDeviceEncoder.
    models = context.encoder._models
    assert models[0] is not None, "first replica missing"
    if len(models) > 1:
        # When multi-GPU configured, extra replicas remain None until
        # a large batch triggers _ensure_pool_ready().
        later_loaded = sum(1 for m in models[1:] if m is not None)
        assert later_loaded == 0, (
            f"expected lazy load, but {later_loaded} extra replicas already loaded"
        )


@then("the device_count matches the configured device list")
def step_device_count(context):
    from atelier.classify import embedding
    assert context.encoder.device_count == len(embedding._devices), (
        f"device_count={context.encoder.device_count} vs _devices={embedding._devices}"
    )
