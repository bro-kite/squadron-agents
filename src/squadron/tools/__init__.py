"""
Pre-built Tool Packs

Domain-specific tool collections that can be easily plugged into agents.
Each pack provides a curated set of tools for common use cases.

Available packs:
- coding: File operations, code search, git integration
- research: Web search, document parsing, summarization
- ops: Shell commands, Docker, monitoring
"""

from squadron.tools.coding import CodingTools
from squadron.tools.research import ResearchTools
from squadron.tools.ops import OpsTools

__all__ = [
    "CodingTools",
    "ResearchTools",
    "OpsTools",
]
