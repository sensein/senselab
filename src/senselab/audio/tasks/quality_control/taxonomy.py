"""Taxonomy node classes and utilities for activity hierarchies."""

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class TaxonomyNode:
    """A node in the activity taxonomy tree.

    This class provides a type-safe, object-oriented interface for navigating
    and manipulating the hierarchical taxonomy structure.

    Attributes:
        name: The name/identifier of this taxonomy node
        metrics: List of metric functions that return numeric/string results
        checks: List of validation functions that return boolean results
        children: Dictionary mapping child names to child TaxonomyNode
            instances
        parent: Reference to parent node (None for root)
    """

    name: str
    metrics: List[Callable] = field(default_factory=list)
    checks: List[Callable] = field(default_factory=list)
    children: Dict[str, "TaxonomyNode"] = field(default_factory=dict)
    parent: Optional["TaxonomyNode"] = field(default=None, repr=False)

    def add_child(self, name: str, node: "TaxonomyNode") -> "TaxonomyNode":
        """Add a child node and establish parent-child relationship.

        Args:
            name: The key name for the child node
            node: The TaxonomyNode instance to add as a child

        Returns:
            The child node that was added
        """
        node.parent = self
        self.children[name] = node
        return node

    def find_path_to(self, target_name: str) -> Optional[List[str]]:
        """Find the path from this node to a descendant with target_name.

        Args:
            target_name: The name of the target node to find

        Returns:
            List of node names representing the path, or None if not found
        """
        if self.name == target_name:
            return [self.name]

        for child_name, child_node in self.children.items():
            child_path = child_node.find_path_to(target_name)
            if child_path:
                return [self.name] + child_path

        return None

    def get_all_evaluations(self) -> List[Callable]:
        """Get all evaluation functions from this node and descendants.

        Returns:
            List of unique evaluation functions (metrics + checks) from
            this node and all its descendants
        """
        evaluations = []
        seen = set()

        # Add this node's evaluations
        for func in self.metrics + self.checks:
            if func not in seen:
                evaluations.append(func)
                seen.add(func)

        # Recursively add children's evaluations
        for child in self.children.values():
            for func in child.get_all_evaluations():
                if func not in seen:
                    evaluations.append(func)
                    seen.add(func)

        return evaluations

    def prune_to_activity(self, activity_name: str) -> Optional["TaxonomyNode"]:
        """Create a pruned copy containing only the path to target activity.

        Args:
            activity_name: Name of the activity to preserve in pruned tree

        Returns:
            New TaxonomyNode representing pruned tree, or None if activity
            not found
        """
        path = self.find_path_to(activity_name)
        if not path:
            return None

        # Create new root with current node's evaluations
        pruned = TaxonomyNode(name=self.name, checks=copy.deepcopy(self.checks), metrics=copy.deepcopy(self.metrics))

        # Build path to target activity
        if len(path) > 1:
            current = pruned
            for i in range(1, len(path)):
                node_name = path[i]
                original_node = self._get_node_by_path(path[: i + 1])
                if original_node:
                    new_node = TaxonomyNode(
                        name=original_node.name,
                        checks=copy.deepcopy(original_node.checks),
                        metrics=copy.deepcopy(original_node.metrics),
                    )
                    current.add_child(node_name, new_node)
                    current = new_node

        return pruned

    def _get_node_by_path(self, path: List[str]) -> Optional["TaxonomyNode"]:
        """Get a node by following a path from this node.

        Args:
            path: List of node names to follow

        Returns:
            The target node, or None if path is invalid
        """
        if not path or path[0] != self.name:
            return None

        if len(path) == 1:
            return self

        next_name = path[1]
        if next_name in self.children:
            next_node = self.children[next_name]
            remaining_path = path[1:]
            return next_node._get_node_by_path(remaining_path)

        return None

    def print_tree(self, prefix: str = "", is_last: bool = True) -> None:
        """Print the taxonomy tree in a nice hierarchical format.

        Args:
            prefix: The prefix string for indentation (used internally)
            is_last: Whether this is the last child at its level
        """
        # Print current node
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{self.name}")

        # Print evaluations if any exist
        evaluation_info = []
        if self.metrics:
            evaluation_info.append(f"metrics: {len(self.metrics)}")
        if self.checks:
            evaluation_info.append(f"checks: {len(self.checks)}")

        if evaluation_info:
            next_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{next_prefix}({', '.join(evaluation_info)})")

        # Print children
        if self.children:
            next_prefix = prefix + ("    " if is_last else "│   ")
            child_items = list(self.children.items())

            for i, (child_name, child_node) in enumerate(child_items):
                is_last_child = i == len(child_items) - 1
                child_node.print_tree(next_prefix, is_last_child)
