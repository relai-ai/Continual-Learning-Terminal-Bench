from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from relai_tbench_agent.config import AGENT_MODEL


class AgentHarness(BaseAgent):
    """Harbor-compatible lazy wrapper for the KIRA-derived agent."""

    SUPPORTS_ATIF = True

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs: Any):
        super().__init__(logs_dir=logs_dir, model_name=AGENT_MODEL, **kwargs)
        from relai_tbench_agent.kira_agent import AgentHarness as KiraAgentHarness

        self._delegate = KiraAgentHarness(logs_dir=logs_dir, model_name=AGENT_MODEL, **kwargs)

    @staticmethod
    def name() -> str:
        return "terminus-kira"

    def version(self) -> str | None:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        await self._delegate.setup(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self._delegate.run(instruction, environment, context)


__all__ = ["AGENT_MODEL", "AgentHarness"]
