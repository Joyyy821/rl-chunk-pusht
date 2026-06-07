"""Agent registry."""

from rl_chunk_pusht.agents.acfql import ACFQLAgent
from rl_chunk_pusht.agents.acrlpd import ACRLPDAgent

agents = {
    "acfql": ACFQLAgent,
    "acrlpd": ACRLPDAgent,
}

__all__ = ["ACFQLAgent", "ACRLPDAgent", "agents"]
