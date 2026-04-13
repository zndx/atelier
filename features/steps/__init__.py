"""Central step re-exports for behave discovery.

Behave discovers steps from features/steps/ only. Domain step definitions
live in <domain>/step_defs/ directories (not steps/, to avoid behave
auto-discovery which would exec them without proper Python import context).
"""

from features.infra.step_defs.config_steps import *  # noqa: F401,F403
from features.infra.step_defs.health_steps import *  # noqa: F401,F403
from features.infra.step_defs.preflight_steps import *  # noqa: F401,F403
from features.deployment.step_defs.runtime_steps import *  # noqa: F401,F403
from features.deployment.step_defs.amp_steps import *  # noqa: F401,F403
from features.deployment.step_defs.naming_steps import *  # noqa: F401,F403
from features.agent.step_defs.agent_steps import *  # noqa: F401,F403
from features.agent.step_defs.classification_steps import *  # noqa: F401,F403
from features.agent.step_defs.bootstrap_steps import *  # noqa: F401,F403
from features.agent.step_defs.backend_steps import *  # noqa: F401,F403
from features.agent.step_defs.synth_steps import *  # noqa: F401,F403
from features.agent.step_defs.ml_steps import *  # noqa: F401,F403
from features.agent.step_defs.ml_e2e_steps import *  # noqa: F401,F403
from features.agent.step_defs.sage_steps import *  # noqa: F401,F403
from features.gateway.step_defs.status_steps import *  # noqa: F401,F403
