"""
Memory Type Definitions

Core data structures for the temporal knowledge graph.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    DOCUMENT = "document"
    CODE = "code"
    TOOL = "tool"
    PREFERENCE = "preference"
    CUSTOM = "custom"


class EdgeType(str, Enum):
    """Types of relationships between entities."""

    # Personal relationships
    KNOWS = "knows"
    WORKS_AT = "works_at"
    LIVES_IN = "lives_in"
    MEMBER_OF = "member_of"
    
    # Preferences
    PREFERS = "prefers"
    DISLIKES = "dislikes"
    USES = "uses"
    
    # Temporal
    HAPPENED_AT = "happened_at"
    CREATED_AT = "created_at"
    MODIFIED_AT = "modified_at"
    
    # Semantic
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    
    # Actions
    MENTIONED = "mentioned"
    DISCUSSED = "discussed"
    REQUESTED = "requested"
    COMPLETED = "completed"
    
    CUSTOM = "custom"


class Entity(BaseModel):
    """
    An entity in the knowledge graph.
    
    Represents a node that can have relationships with other entities.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: EntityType = EntityType.CUSTOM
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    
    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    
    # Source tracking
    source_session_id: UUID | None = None
    source_message_id: UUID | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True

    @property
    def is_valid(self) -> bool:
        """Check if the entity is currently valid."""
        now = datetime.utcnow()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True


class Edge(BaseModel):
    """
    A relationship between two entities.
    
    Edges are temporal - they can be invalidated when facts change.
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    edge_type: EdgeType = EdgeType.RELATED_TO
    properties: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, ge=0.0)
    
    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None
    
    # Invalidation tracking
    invalidated: bool = False
    invalidated_at: datetime | None = None
    invalidated_by: UUID | None = None  # ID of the edge that replaced this one
    invalidation_reason: str | None = None
    
    # Source tracking
    source_session_id: UUID | None = None
    source_message_id: UUID | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True

    @property
    def is_valid(self) -> bool:
        """Check if the edge is currently valid."""
        if self.invalidated:
            return False
        now = datetime.utcnow()
        if now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def invalidate(
        self,
        reason: str,
        replaced_by: UUID | None = None,
    ) -> "Edge":
        """Create an invalidated version of this edge."""
        return self.model_copy(
            update={
                "invalidated": True,
                "invalidated_at": datetime.utcnow(),
                "invalidated_by": replaced_by,
                "invalidation_reason": reason,
                "valid_until": datetime.utcnow(),
            }
        )


class Fact(BaseModel):
    """
    A fact extracted from conversation.
    
    Facts are the atomic units of knowledge that get stored in the graph.
    They represent a subject-predicate-object triple with temporal context.
    """

    id: UUID = Field(default_factory=uuid4)
    subject: str
    predicate: str
    object: str
    
    # Structured references
    subject_entity_id: UUID | None = None
    object_entity_id: UUID | None = None
    edge_id: UUID | None = None
    
    # Context
    raw_text: str  # Original text this fact was extracted from
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Temporal
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None
    
    # Source
    session_id: UUID | None = None
    message_id: UUID | None = None

    class Config:
        frozen = True

    def to_triple(self) -> tuple[str, str, str]:
        """Return the fact as a (subject, predicate, object) triple."""
        return (self.subject, self.predicate, self.object)

    def to_sentence(self) -> str:
        """Convert the fact to a natural language sentence."""
        return f"{self.subject} {self.predicate} {self.object}"


class MemoryQuery(BaseModel):
    """Query parameters for memory retrieval."""

    query: str
    session_id: str | None = None
    entity_types: list[EntityType] | None = None
    edge_types: list[EdgeType] | None = None
    
    # Temporal filters
    valid_at: datetime | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    
    # Result limits
    max_entities: int = Field(default=10, ge=1, le=100)
    max_edges: int = Field(default=20, ge=1, le=200)
    max_facts: int = Field(default=10, ge=1, le=100)
    
    # Similarity threshold
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Include invalidated items
    include_invalidated: bool = False


class MemoryResult(BaseModel):
    """Result from a memory query."""

    query: MemoryQuery
    entities: list[Entity] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    
    # Metadata
    query_time_ms: float = 0.0
    total_entities_searched: int = 0
    total_edges_searched: int = 0

    def to_context_dict(self) -> dict[str, Any]:
        """Convert to a context dictionary for agent state."""
        return {
            "facts": [f.to_sentence() for f in self.facts],
            "entities": [
                {"name": e.name, "type": e.entity_type, "properties": e.properties}
                for e in self.entities
            ],
            "relationships": [
                {
                    "source": str(e.source_id),
                    "target": str(e.target_id),
                    "type": e.edge_type,
                }
                for e in self.edges
            ],
        }

    def to_prompt_context(self) -> str:
        """Convert to a string suitable for inclusion in prompts."""
        lines = ["## Relevant Memory Context\n"]
        
        if self.facts:
            lines.append("### Known Facts:")
            for fact in self.facts:
                lines.append(f"- {fact.to_sentence()}")
            lines.append("")
        
        if self.entities:
            lines.append("### Relevant Entities:")
            for entity in self.entities:
                props = ", ".join(f"{k}={v}" for k, v in entity.properties.items())
                lines.append(f"- {entity.name} ({entity.entity_type}): {props}")
            lines.append("")
        
        return "\n".join(lines)