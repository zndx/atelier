"""Tests for base client: URL resolution and HTTP plumbing."""

import pytest

from governance.client import CDPUrlResolver, ClientConfig, BaseClient, GovernanceClient


class TestCDPUrlResolver:
    """Verify URL normalisation for Atlas and Ranger across CDP deployment patterns."""

    def test_atlas_bare_host(self):
        ep = CDPUrlResolver.resolve_atlas("https://atlas-host:21000")
        assert ep.url == "https://atlas-host:21000/api/atlas/v2"

    def test_atlas_cdp_proxy(self):
        ep = CDPUrlResolver.resolve_atlas("https://cdp.example.com/cdp-proxy-api/atlas")
        assert ep.url == "https://cdp.example.com/cdp-proxy-api/atlas/api/atlas/v2"

    def test_atlas_explicit_prefix(self):
        ep = CDPUrlResolver.resolve_atlas("https://host:21000/api/atlas/v2")
        assert ep.url == "https://host:21000/api/atlas/v2"

    def test_atlas_trailing_slash(self):
        ep = CDPUrlResolver.resolve_atlas("https://host:21000/")
        assert ep.url == "https://host:21000/api/atlas/v2"

    def test_ranger_bare_host(self):
        ep = CDPUrlResolver.resolve_ranger("https://ranger-host:6182")
        assert ep.url == "https://ranger-host:6182/service/public/v2/api"

    def test_ranger_cdp_proxy(self):
        ep = CDPUrlResolver.resolve_ranger("https://cdp.example.com/cdp-proxy-api/ranger")
        assert ep.url == "https://cdp.example.com/cdp-proxy-api/ranger/service/public/v2/api"

    def test_ranger_explicit_prefix(self):
        ep = CDPUrlResolver.resolve_ranger("https://host:6182/service/public/v2/api")
        assert ep.url == "https://host:6182/service/public/v2/api"


class TestGovernanceClient:
    """GovernanceClient facade wiring."""

    def test_atlas_only(self):
        gc = GovernanceClient(atlas_url="https://host:21000", username="u", password="p")
        assert gc.atlas is not None
        with pytest.raises(RuntimeError, match="ranger_url"):
            _ = gc.ranger

    def test_ranger_only(self):
        gc = GovernanceClient(ranger_url="https://host:6182", username="u", password="p")
        assert gc.ranger is not None
        with pytest.raises(RuntimeError, match="atlas_url"):
            _ = gc.atlas

    def test_both(self):
        gc = GovernanceClient(
            atlas_url="https://host:21000",
            ranger_url="https://host:6182",
            username="u",
            password="p",
        )
        assert gc.atlas is not None
        assert gc.ranger is not None
