"""GitHub App integration for overwatch PR creation.

Uses JWT authentication (App private key → installation token → API calls)
for headless operation on CAI. The ``gh`` CLI is available in devenv but
the App approach is needed for automated/CI contexts.

Usage::

    from atelier.overwatch.github_app import GitHubApp

    app = GitHubApp(
        app_id="12345",
        private_key_path="/path/to/private-key.pem",
        repo="zndx/atelier",
    )
    token = app.get_installation_token()
    app.create_branch("trunk", "overwatch/fix-batch-size")
    app.create_pr(
        head="overwatch/fix-batch-size",
        base="trunk",
        title="fix: reduce batch size for large vocabularies",
        body="Overwatch detected truncation in 3 consecutive runs...",
    )
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)


class GitHubApp:
    """GitHub App client for overwatch PR workflows.

    Autonomy levels:
    - ``monitor``: no GitHub operations (read-only observation)
    - ``propose``: create branches + draft PRs (never merges)
    - ``autonomous``: create + merge PRs (future, not yet implemented)
    """

    API_BASE = "https://api.github.com"

    def __init__(
        self,
        app_id: str,
        private_key_path: str,
        repo: str,
        *,
        autonomy: str = "propose",
    ) -> None:
        self._app_id = app_id
        self._private_key = Path(private_key_path).read_text() if private_key_path else ""
        self._repo = repo  # "owner/repo"
        self._autonomy = autonomy
        self._installation_token: str | None = None
        self._token_expires_at: float = 0.0

    @classmethod
    def from_config(cls, cfg) -> GitHubApp | None:
        """Build from AtelierConfig. Returns None if not configured."""
        if not cfg.overwatch_github_app_id or not cfg.overwatch_github_repo:
            return None
        return cls(
            app_id=cfg.overwatch_github_app_id,
            private_key_path=cfg.overwatch_github_private_key_path,
            repo=cfg.overwatch_github_repo,
            autonomy=cfg.overwatch_autonomy,
        )

    @property
    def configured(self) -> bool:
        return bool(self._app_id and self._private_key and self._repo)

    # ── Authentication ───────────────────────────────────────────

    def _make_jwt(self) -> str:
        """Create a short-lived JWT signed with the App's private key."""
        try:
            import jwt  # PyJWT
        except ImportError:
            raise ImportError(
                "PyJWT required for GitHub App auth: pip install PyJWT"
            )
        now = int(time.time())
        payload = {
            "iat": now - 60,  # issued at (clock drift buffer)
            "exp": now + (10 * 60),  # 10 min max for App JWTs
            "iss": self._app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    def get_installation_token(self) -> str:
        """Get or refresh the installation access token."""
        if self._installation_token and time.time() < self._token_expires_at:
            return self._installation_token

        jwt_token = self._make_jwt()
        # Find the installation for our repo
        resp = requests.get(
            f"{self.API_BASE}/app/installations",
            headers={"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"},
            timeout=15,
        )
        resp.raise_for_status()
        installations = resp.json()

        if not installations:
            raise RuntimeError("No installations found for this GitHub App")

        # Use first installation (or find the one matching our repo owner)
        owner = self._repo.split("/")[0]
        install = next(
            (i for i in installations if i.get("account", {}).get("login") == owner),
            installations[0],
        )
        install_id = install["id"]

        # Create installation token
        resp = requests.post(
            f"{self.API_BASE}/app/installations/{install_id}/access_tokens",
            headers={"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._installation_token = data["token"]
        # Token expires in ~1 hour; refresh at 50 min
        self._token_expires_at = time.time() + 3000
        return self._installation_token

    def _headers(self) -> dict[str, str]:
        token = self.get_installation_token()
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

    # ── Branch operations ────────────────────────────────────────

    def create_branch(self, base: str, name: str) -> str:
        """Create a new branch from base. Returns the branch name."""
        if self._autonomy == "monitor":
            raise RuntimeError("Cannot create branches in monitor-only mode")

        # Get base ref SHA
        resp = requests.get(
            f"{self.API_BASE}/repos/{self._repo}/git/ref/heads/{base}",
            headers=self._headers(), timeout=15,
        )
        resp.raise_for_status()
        sha = resp.json()["object"]["sha"]

        # Create branch ref
        resp = requests.post(
            f"{self.API_BASE}/repos/{self._repo}/git/refs",
            headers=self._headers(), timeout=15,
            json={"ref": f"refs/heads/{name}", "sha": sha},
        )
        if resp.status_code == 422:
            log.info("Branch %s already exists", name)
            return name
        resp.raise_for_status()
        log.info("Created branch %s from %s (%s)", name, base, sha[:8])
        return name

    # ── PR operations ────────────────────────────────────────────

    def create_pr(
        self,
        head: str,
        base: str,
        title: str,
        body: str,
        *,
        draft: bool = True,
    ) -> dict[str, Any]:
        """Create a pull request. Draft by default (propose mode)."""
        if self._autonomy == "monitor":
            raise RuntimeError("Cannot create PRs in monitor-only mode")

        resp = requests.post(
            f"{self.API_BASE}/repos/{self._repo}/pulls",
            headers=self._headers(), timeout=15,
            json={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        )
        resp.raise_for_status()
        pr = resp.json()
        log.info("Created PR #%d: %s", pr["number"], title)
        return pr

    # ── Health probe ─────────────────────────────────────────────

    def ping(self) -> dict[str, Any]:
        """Verify GitHub App connectivity."""
        if not self.configured:
            return {"ok": False, "error": "Not configured (missing app_id, key, or repo)"}
        try:
            token = self.get_installation_token()
            return {"ok": True, "repo": self._repo, "autonomy": self._autonomy}
        except Exception as e:
            return {"ok": False, "error": str(e)}
