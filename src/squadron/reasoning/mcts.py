"""
Monte Carlo Tree Search (MCTS) Controller

Generic MCTS implementation for tree-search reasoning.
Used by LATS for exploring the space of possible actions.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")  # State type


@dataclass
class MCTSNode(Generic[T]):
    """
    A node in the MCTS tree.
    
    Represents a state in the search space with associated
    statistics for UCB selection.
    """

    id: UUID = field(default_factory=uuid4)
    state: T | None = None
    parent_id: UUID | None = None
    children_ids: list[UUID] = field(default_factory=list)
    
    # Action that led to this state
    action: Any = None
    action_description: str = ""
    
    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0
    
    # Node metadata
    depth: int = 0
    is_terminal: bool = False
    is_expanded: bool = False
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def ucb_score(self) -> float:
        """
        Upper Confidence Bound score for selection.
        
        UCB1 formula: value + C * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float("inf")  # Unexplored nodes have infinite priority
        return self.value  # Parent visits handled in selection

    def __hash__(self) -> int:
        return hash(self.id)


class MCTSController(Generic[T]):
    """
    Monte Carlo Tree Search controller.
    
    Implements the four phases of MCTS:
    1. Selection: Choose promising nodes using UCB
    2. Expansion: Add new child nodes
    3. Simulation: Evaluate node value
    4. Backpropagation: Update ancestor statistics
    
    Example:
        ```python
        mcts = MCTSController(
            expand_fn=generate_actions,
            simulate_fn=evaluate_action,
            exploration_constant=1.414,
        )
        
        best_action = await mcts.search(
            initial_state=state,
            budget=100,
        )
        ```
    """

    def __init__(
        self,
        expand_fn: Callable[[T], list[tuple[Any, str, T]]],
        simulate_fn: Callable[[T], float],
        is_terminal_fn: Callable[[T], bool] | None = None,
        exploration_constant: float = 1.414,  # sqrt(2)
        max_depth: int = 10,
    ):
        """
        Initialize the MCTS controller.
        
        Args:
            expand_fn: Function that generates (action, description, new_state) tuples
            simulate_fn: Function that evaluates a state and returns a value [0, 1]
            is_terminal_fn: Function that checks if a state is terminal
            exploration_constant: UCB exploration parameter (C)
            max_depth: Maximum tree depth
        """
        self.expand_fn = expand_fn
        self.simulate_fn = simulate_fn
        self.is_terminal_fn = is_terminal_fn or (lambda s: False)
        self.exploration_constant = exploration_constant
        self.max_depth = max_depth
        
        # Tree storage
        self._nodes: dict[UUID, MCTSNode[T]] = {}
        self._root_id: UUID | None = None

    def reset(self) -> None:
        """Reset the tree for a new search."""
        self._nodes.clear()
        self._root_id = None

    async def search(
        self,
        initial_state: T,
        budget: int = 100,
    ) -> tuple[Any, list[MCTSNode[T]]]:
        """
        Perform MCTS search from the initial state.
        
        Args:
            initial_state: Starting state
            budget: Number of simulations to run
            
        Returns:
            Tuple of (best_action, trajectory)
        """
        self.reset()
        
        # Create root node
        root = MCTSNode(
            state=initial_state,
            depth=0,
            is_terminal=self.is_terminal_fn(initial_state),
        )
        self._nodes[root.id] = root
        self._root_id = root.id
        
        logger.debug("Starting MCTS search", budget=budget)
        
        # Run simulations
        for i in range(budget):
            # Selection
            node = await self._select(root)
            
            # Expansion (if not terminal and not at max depth)
            if not node.is_terminal and node.depth < self.max_depth:
                node = await self._expand(node)
            
            # Simulation
            value = await self._simulate(node)
            
            # Backpropagation
            await self._backpropagate(node, value)
            
            if (i + 1) % 20 == 0:
                logger.debug(
                    "MCTS progress",
                    iteration=i + 1,
                    nodes=len(self._nodes),
                    root_visits=root.visits,
                )
        
        # Select best action from root
        best_child = self._select_best_child(root, exploration=False)
        
        if best_child is None:
            logger.warning("No valid actions found")
            return None, [root]
        
        # Build trajectory
        trajectory = self._build_trajectory(best_child)
        
        logger.info(
            "MCTS search complete",
            best_action=best_child.action_description,
            best_value=best_child.value,
            total_nodes=len(self._nodes),
        )
        
        return best_child.action, trajectory

    async def _select(self, node: MCTSNode[T]) -> MCTSNode[T]:
        """
        Selection phase: Traverse tree using UCB until reaching a leaf.
        """
        current = node
        
        while current.is_expanded and current.children_ids:
            best_child = self._select_best_child(current, exploration=True)
            if best_child is None:
                break
            current = best_child
        
        return current

    def _select_best_child(
        self,
        node: MCTSNode[T],
        exploration: bool = True,
    ) -> MCTSNode[T] | None:
        """
        Select the best child using UCB1.
        
        Args:
            node: Parent node
            exploration: Whether to include exploration bonus
            
        Returns:
            Best child node or None
        """
        if not node.children_ids:
            return None
        
        best_score = float("-inf")
        best_child = None
        
        for child_id in node.children_ids:
            child = self._nodes.get(child_id)
            if child is None:
                continue
            
            if exploration:
                # UCB1 formula
                if child.visits == 0:
                    score = float("inf")
                else:
                    exploitation = child.value
                    exploration_bonus = self.exploration_constant * math.sqrt(
                        math.log(node.visits + 1) / child.visits
                    )
                    score = exploitation + exploration_bonus
            else:
                # Pure exploitation for final selection
                score = child.value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    async def _expand(self, node: MCTSNode[T]) -> MCTSNode[T]:
        """
        Expansion phase: Add child nodes for unexplored actions.
        """
        if node.state is None:
            return node
        
        # Generate possible actions
        try:
            expansions = self.expand_fn(node.state)
        except Exception as e:
            logger.error("Expansion failed", error=str(e))
            return node
        
        if not expansions:
            node.is_terminal = True
            return node
        
        # Create child nodes
        for action, description, new_state in expansions:
            child = MCTSNode(
                state=new_state,
                parent_id=node.id,
                action=action,
                action_description=description,
                depth=node.depth + 1,
                is_terminal=self.is_terminal_fn(new_state),
            )
            self._nodes[child.id] = child
            node.children_ids.append(child.id)
        
        node.is_expanded = True
        
        # Return a random unexplored child for simulation
        unexplored = [
            self._nodes[cid]
            for cid in node.children_ids
            if self._nodes[cid].visits == 0
        ]
        
        if unexplored:
            return random.choice(unexplored)
        
        return node

    async def _simulate(self, node: MCTSNode[T]) -> float:
        """
        Simulation phase: Evaluate the node's state.
        
        Returns a value in [0, 1] representing the quality of the state.
        """
        if node.state is None:
            return 0.0
        
        try:
            value = self.simulate_fn(node.state)
            # Clamp to [0, 1]
            return max(0.0, min(1.0, value))
        except Exception as e:
            logger.error("Simulation failed", error=str(e))
            return 0.0

    async def _backpropagate(self, node: MCTSNode[T], value: float) -> None:
        """
        Backpropagation phase: Update statistics up the tree.
        """
        current: MCTSNode[T] | None = node
        
        while current is not None:
            current.visits += 1
            current.total_value += value
            
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                current = None

    def _build_trajectory(self, node: MCTSNode[T]) -> list[MCTSNode[T]]:
        """Build the trajectory from root to the given node."""
        trajectory = []
        current: MCTSNode[T] | None = node
        
        while current is not None:
            trajectory.append(current)
            if current.parent_id:
                current = self._nodes.get(current.parent_id)
            else:
                current = None
        
        trajectory.reverse()
        return trajectory

    def get_top_actions(
        self,
        n: int = 5,
    ) -> list[tuple[Any, str, float, int]]:
        """
        Get the top N actions from the root.
        
        Returns:
            List of (action, description, value, visits) tuples
        """
        if self._root_id is None:
            return []
        
        root = self._nodes.get(self._root_id)
        if root is None or not root.children_ids:
            return []
        
        children = [
            self._nodes[cid]
            for cid in root.children_ids
            if cid in self._nodes
        ]
        
        # Sort by value (descending)
        children.sort(key=lambda c: c.value, reverse=True)
        
        return [
            (c.action, c.action_description, c.value, c.visits)
            for c in children[:n]
        ]

    @property
    def tree_stats(self) -> dict[str, Any]:
        """Get statistics about the current tree."""
        if not self._nodes:
            return {"nodes": 0, "max_depth": 0, "avg_branching": 0}
        
        depths = [n.depth for n in self._nodes.values()]
        branching = [
            len(n.children_ids)
            for n in self._nodes.values()
            if n.children_ids
        ]
        
        return {
            "nodes": len(self._nodes),
            "max_depth": max(depths) if depths else 0,
            "avg_branching": sum(branching) / len(branching) if branching else 0,
            "root_visits": self._nodes[self._root_id].visits if self._root_id else 0,
        }