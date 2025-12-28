"""
Configuration management for Squadron Agent Framework.

Uses pydantic-settings for environment variable loading and validation.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for LLM providers."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: Literal["openai", "anthropic", "ollama", "huggingface", "openai_compatible"] = Field(
        default="openai",
        description="LLM provider to use",
    )
    model: str = Field(
        default="gpt-4o",
        description="Model name to use",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in response",
    )
    
    # API Keys
    openai_api_key: SecretStr | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        alias="ANTHROPIC_API_KEY",
    )
    huggingface_api_key: SecretStr | None = Field(
        default=None,
        alias="HF_TOKEN",
    )
    
    # Custom endpoint (for OpenAI-compatible APIs, vLLM, DigitalOcean, etc.)
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL for OpenAI-compatible endpoints",
    )
    
    # Ollama settings
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    
    # Hugging Face settings
    use_local_model: bool = Field(
        default=False,
        description="Use local transformers model instead of API",
    )


class MemoryConfig(BaseSettings):
    """Configuration for Graphiti memory system."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        alias="NEO4J_URI",
        description="Neo4j connection URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        alias="NEO4J_USER",
        description="Neo4j username",
    )
    # Security: No default password - must be explicitly configured
    neo4j_password: SecretStr | None = Field(
        default=None,
        alias="NEO4J_PASSWORD",
        description="Neo4j password (required for production)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model for vector similarity",
    )
    embedding_dim: int = Field(
        default=1536,
        description="Embedding dimension",
    )

    def validate_credentials(self) -> None:
        """Validate that required credentials are configured."""
        if self.neo4j_password is None:
            raise ValueError(
                "NEO4J_PASSWORD must be configured. "
                "Set the NEO4J_PASSWORD environment variable."
            )


class ReasoningConfig(BaseSettings):
    """Configuration for LATS reasoning engine."""

    model_config = SettingsConfigDict(env_prefix="REASONING_")

    n_candidates: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate actions to generate",
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        description="Maximum tree search depth",
    )
    exploration_constant: float = Field(
        default=1.414,
        gt=0,
        description="UCB exploration constant (sqrt(2) is standard)",
    )
    simulation_budget: int = Field(
        default=100,
        ge=1,
        description="Maximum simulations per decision",
    )
    verifier_model: str = Field(
        default="gpt-4o-mini",
        description="Model for list-wise verification",
    )


class MCPConfig(BaseSettings):
    """Configuration for MCP connectivity."""

    model_config = SettingsConfigDict(env_prefix="MCP_")

    servers_config_path: str = Field(
        default="mcp_servers.json",
        description="Path to MCP servers configuration file",
    )
    connection_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Connection timeout in seconds",
    )
    request_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Request timeout in seconds",
    )


class GovernanceConfig(BaseSettings):
    """Configuration for evaluation and safety."""

    model_config = SettingsConfigDict(env_prefix="GOVERNANCE_")

    enable_guardrails: bool = Field(
        default=True,
        description="Enable safety guardrails",
    )
    require_human_approval: list[str] = Field(
        default_factory=lambda: ["delete_file", "transfer_money", "execute_code"],
        description="Tools requiring human approval",
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        description="Maximum agent loop iterations",
    )
    eval_on_completion: bool = Field(
        default=True,
        description="Run evaluation on task completion",
    )


class SkillsConfig(BaseSettings):
    """Configuration for Agent Skills support."""

    model_config = SettingsConfigDict(env_prefix="SKILLS_")

    enabled: bool = Field(
        default=True,
        description="Enable Agent Skills support",
    )
    directories: list[str] = Field(
        default_factory=lambda: [
            ".github/skills",
            ".claude/skills",
            ".squadron/skills",
            "skills",
        ],
        description="Directories to search for skills (relative to workspace)",
    )
    max_skills_in_context: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum skills to include in LLM context",
    )
    relevance_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to include a skill",
    )
    auto_discover: bool = Field(
        default=True,
        description="Automatically discover skills on agent initialization",
    )


class EvolutionConfig(BaseSettings):
    """Configuration for self-improvement system."""

    model_config = SettingsConfigDict(env_prefix="EVOLUTION_")

    enable_self_improvement: bool = Field(
        default=False,
        description="Enable self-improvement capabilities",
    )
    sandbox_image: str = Field(
        default="squadron-sandbox:latest",
        description="Docker image for sandbox execution",
    )
    mutation_rate: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Prompt mutation rate for ADAS",
    )
    min_improvement_threshold: float = Field(
        default=0.05,
        ge=0,
        description="Minimum improvement to accept mutation",
    )


class SquadronConfig(BaseSettings):
    """
    Main configuration for Squadron Agent Framework.
    
    Aggregates all layer-specific configurations.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Agent identity
    agent_name: str = Field(
        default="squadron",
        description="Agent name for identification",
    )
    agent_version: str = Field(
        default="0.1.0",
        description="Agent version",
    )

    # Layer configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )

    @classmethod
    def from_env(cls) -> "SquadronConfig":
        """Load configuration from environment variables."""
        return cls()

    @classmethod
    def from_file(cls, path: str) -> "SquadronConfig":
        """Load configuration from a file."""
        import json
        from pathlib import Path

        config_path = Path(path)
        if config_path.suffix == ".json":
            with open(config_path) as f:
                data = json.load(f)
            return cls(**data)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")