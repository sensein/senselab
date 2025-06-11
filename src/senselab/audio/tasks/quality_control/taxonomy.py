"""Taxonomy node classes and utilities for bioacoustic activity hierarchies."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class TaxonomyNode:
    """A node in the bioacoustic activity taxonomy tree.

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
        """Find the path through the taxonomy tree from this node to a target node by name.

        Args:
            target_name: The name of the node to find in the taxonomy tree

        Returns:
            List of node names representing the path through the taxonomy tree from this node
            to the target node, or None if target not found in this subtree
        """
        # Base case: current node matches target
        if self.name == target_name:
            return [self.name]

        # Recursively search children
        for child_name, child in self.children.items():
            child_path = child.find_path_to(target_name)
            # Only return path if it leads to target_name
            if child_path is not None and child_path[-1] == target_name:
                return [self.name] + child_path
        return None

    def get_all_evaluations(self) -> Sequence[Callable]:
        """Recursively collect all evaluation functions from this subtree.

        Returns:
            Sequence of all unique evaluation functions (checks + metrics)
            from this node and all descendants
        """
        evaluations = []

        # Add current node's evaluations
        for func in self.metrics + self.checks:
            if func not in evaluations:
                evaluations.append(func)

        # Recursively add children's evaluations
        for child in self.children.values():
            for func in child.get_all_evaluations():
                if func not in evaluations:
                    evaluations.append(func)

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
        pruned = TaxonomyNode(name=self.name, checks=self.checks.copy(), metrics=self.metrics.copy())

        # Build path to target activity
        if len(path) > 1:
            current = pruned
            for i in range(1, len(path)):
                node_name = path[i]
                original_node = self._get_node_by_path(path[: i + 1])
                if original_node:
                    new_node = TaxonomyNode(
                        name=original_node.name,
                        checks=original_node.checks.copy(),
                        metrics=original_node.metrics.copy(),
                    )
                    current.add_child(node_name, new_node)
                    current = new_node

        return pruned

    def _get_node_by_path(self, path: List[str]) -> Optional["TaxonomyNode"]:
        """Get node at the specified path from this node.

        Args:
            path: List of node names representing path from this node

        Returns:
            TaxonomyNode at the path, or None if path invalid
        """
        if not path or path[0] != self.name:
            return None

        if len(path) == 1:
            return self

        next_name = path[1]
        if next_name in self.children:
            return self.children[next_name]._get_node_by_path(path[1:])

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert this taxonomy tree to the legacy dictionary format.

        Returns:
            Dictionary representation compatible with existing code
        """
        result: Dict[str, Any] = {
            "checks": self.checks.copy(),
            "metrics": self.metrics.copy(),
            "subclass": None,
        }

        if self.children:
            subclass_dict: Dict[str, Any] = {}
            for child_name, child_node in self.children.items():
                child_dict = child_node.to_dict()
                # Extract the inner dictionary since to_dict() wraps with name
                subclass_dict[child_name] = child_dict[child_node.name]
            result["subclass"] = subclass_dict

        return {self.name: result}


def build_taxonomy_tree_from_dict(taxonomy_dict: Dict[str, Any], name: Optional[str] = None) -> TaxonomyNode:
    """Build TaxonomyNode tree from legacy dictionary format.

    Args:
        taxonomy_dict: Dictionary in the legacy format
        name: Name for the root node (inferred if None)

    Returns:
        TaxonomyNode representing the taxonomy tree
    """
    if name is None:
        # Root level - get the first key as root name
        root_name = list(taxonomy_dict.keys())[0]
        root_data = taxonomy_dict[root_name]
    else:
        root_name = name
        root_data = taxonomy_dict

    # Create root node
    root = TaxonomyNode(
        name=root_name,
        checks=root_data.get("checks", []).copy(),
        metrics=root_data.get("metrics", []).copy(),
    )

    # Recursively add children
    subclass = root_data.get("subclass")
    if isinstance(subclass, dict):
        for child_name, child_data in subclass.items():
            child_node = build_taxonomy_tree_from_dict(child_data, child_name)
            root.add_child(child_name, child_node)

    return root


def dict_to_taxonomy_tree(taxonomy_dict: Dict[str, Any]) -> TaxonomyNode:
    """Convenience function to convert dictionary to TaxonomyNode tree.

    Args:
        taxonomy_dict: Dictionary in the legacy format

    Returns:
        TaxonomyNode representing the taxonomy tree
    """
    return build_taxonomy_tree_from_dict(taxonomy_dict)


def taxonomy_tree_to_dict(taxonomy_node: TaxonomyNode) -> Dict[str, Any]:
    """Convenience function to convert TaxonomyNode tree to dictionary.

    Args:
        taxonomy_node: Root node of the taxonomy tree

    Returns:
        Dictionary representation compatible with existing code
    """
    return taxonomy_node.to_dict()
