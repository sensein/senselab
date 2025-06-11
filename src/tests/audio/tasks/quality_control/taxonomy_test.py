"""Tests for the taxonomy module."""

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.taxonomy import (
    TaxonomyNode,
)


# Mock evaluation functions for testing
def mock_check_function(audio: Audio) -> bool:
    """Mock check function for testing."""
    return True


def mock_metric_function(audio: Audio) -> float:
    """Mock metric function for testing."""
    return 0.5


def mock_check_function_2(audio: Audio) -> bool:
    """Second mock check function for testing."""
    return False


def mock_metric_function_2(audio: Audio) -> float:
    """Second mock metric function for testing."""
    return 1.0


def mock_check_function_3(audio: Audio) -> bool:
    """Third mock check function for testing."""
    return True


def mock_metric_function_3(audio: Audio) -> float:
    """Third mock metric function for testing."""
    return 2.0


def test_add_child_basic() -> None:
    """Test basic functionality of adding a child node."""
    parent = TaxonomyNode(name="parent")
    child = TaxonomyNode(name="child")

    result = parent.add_child("child", child)

    # Check return value
    assert result is child, "add_child should return the child node"

    # Check parent-child relationship
    assert "child" in parent.children, "Child should be in parent's children dict"
    assert parent.children["child"] is child, "Child should be accessible by name"
    assert child.parent is parent, "Child's parent should be set correctly"


def test_add_child_with_evaluations() -> None:
    """Test adding a child node that has its own evaluations."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])
    child = TaxonomyNode(name="child", checks=[mock_check_function_2], metrics=[mock_metric_function_2])

    parent.add_child("child", child)

    # Verify evaluations are preserved
    assert child.checks == [mock_check_function_2]
    assert child.metrics == [mock_metric_function_2]
    assert parent.checks == [mock_check_function]
    assert parent.metrics == [mock_metric_function]


def test_add_multiple_children() -> None:
    """Test adding multiple children to a parent node."""
    parent = TaxonomyNode(name="parent")
    child1 = TaxonomyNode(name="child1")
    child2 = TaxonomyNode(name="child2")
    child3 = TaxonomyNode(name="child3")

    parent.add_child("child1", child1)
    parent.add_child("child2", child2)
    parent.add_child("child3", child3)

    # Check all children are added
    assert len(parent.children) == 3
    assert "child1" in parent.children
    assert "child2" in parent.children
    assert "child3" in parent.children

    # Check parent relationships
    assert child1.parent is parent
    assert child2.parent is parent
    assert child3.parent is parent


def test_add_child_overwrites_existing() -> None:
    """Test adding a child with existing key overwrites the previous child."""
    parent = TaxonomyNode(name="parent")
    child1 = TaxonomyNode(name="original_child")
    child2 = TaxonomyNode(name="new_child")

    parent.add_child("child", child1)
    assert parent.children["child"] is child1
    assert child1.parent is parent

    # Overwrite with new child
    parent.add_child("child", child2)
    assert parent.children["child"] is child2
    assert child2.parent is parent
    # Note: child1.parent might still reference parent in this implementation


def test_add_child_nested_hierarchy() -> None:
    """Test creating a nested hierarchy using add_child."""
    root = TaxonomyNode(name="root")
    level1 = TaxonomyNode(name="level1")
    level2 = TaxonomyNode(name="level2")
    leaf = TaxonomyNode(name="leaf")

    root.add_child("level1", level1)
    level1.add_child("level2", level2)
    level2.add_child("leaf", leaf)

    # Test the full hierarchy
    assert root.children["level1"] is level1
    assert level1.parent is root
    assert level1.children["level2"] is level2
    assert level2.parent is level1
    assert level2.children["leaf"] is leaf
    assert leaf.parent is level2
    assert len(leaf.children) == 0


def test_add_child_maintains_original_node_properties() -> None:
    """Test that add_child preserves all original node properties."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])

    # Child with its own properties
    child = TaxonomyNode(name="child", checks=[mock_check_function_2], metrics=[mock_metric_function_2])

    # Add an existing child to the child node before adding to parent
    grandchild = TaxonomyNode(name="grandchild")
    child.add_child("grandchild", grandchild)

    # Now add child to parent
    parent.add_child("child", child)

    # Verify all properties are maintained
    assert child.name == "child"
    assert child.checks == [mock_check_function_2]
    assert child.metrics == [mock_metric_function_2]
    assert "grandchild" in child.children
    assert child.children["grandchild"] is grandchild
    assert grandchild.parent is child


def test_add_child_chaining() -> None:
    """Test that add_child returns allow for method chaining."""
    parent = TaxonomyNode(name="parent")
    child = TaxonomyNode(name="child")

    # Test that we can chain operations on the returned child
    result = parent.add_child("child", child)
    grandchild = TaxonomyNode(name="grandchild")

    # This should work if add_child returns the child node
    result.add_child("grandchild", grandchild)

    assert parent.children["child"].children["grandchild"] is grandchild
    assert grandchild.parent is child


def test_find_path_to_self() -> None:
    """Test finding path to the current node itself."""
    node = TaxonomyNode(name="test_node")

    path = node.find_path_to("test_node")

    assert path == ["test_node"]


def test_find_path_to_direct_child() -> None:
    """Test finding path to a direct child node."""
    parent = TaxonomyNode(name="parent")
    child = TaxonomyNode(name="child")
    parent.add_child("child", child)

    path = parent.find_path_to("child")

    assert path == ["parent", "child"]


def test_find_path_to_nested_child() -> None:
    """Test finding path through multiple levels of hierarchy."""
    root = TaxonomyNode(name="root")
    level1 = TaxonomyNode(name="level1")
    level2 = TaxonomyNode(name="level2")
    leaf = TaxonomyNode(name="leaf")

    root.add_child("level1", level1)
    level1.add_child("level2", level2)
    level2.add_child("leaf", leaf)

    path = root.find_path_to("leaf")

    assert path == ["root", "level1", "level2", "leaf"]


def test_find_path_to_nonexistent_node() -> None:
    """Test finding path to a node that doesn't exist."""
    parent = TaxonomyNode(name="parent")
    child = TaxonomyNode(name="child")
    parent.add_child("child", child)

    path = parent.find_path_to("nonexistent")

    assert path is None


def test_find_path_to_empty_tree() -> None:
    """Test finding path in a tree with no children."""
    root = TaxonomyNode(name="root")

    # Try to find a non-existent child in empty tree
    path = root.find_path_to("nonexistent")
    assert path is None

    # Finding self should still work
    path = root.find_path_to("root")
    assert path == ["root"]


def test_find_path_to_multiple_children() -> None:
    """Test finding paths in a tree with multiple children at same level."""
    parent = TaxonomyNode(name="parent")
    child1 = TaxonomyNode(name="child1")
    child2 = TaxonomyNode(name="child2")
    child3 = TaxonomyNode(name="child3")

    parent.add_child("child1", child1)
    parent.add_child("child2", child2)
    parent.add_child("child3", child3)

    # Test finding each child
    assert parent.find_path_to("child1") == ["parent", "child1"]
    assert parent.find_path_to("child2") == ["parent", "child2"]
    assert parent.find_path_to("child3") == ["parent", "child3"]


def test_find_path_to_complex_hierarchy() -> None:
    """Test finding paths in a complex multi-branch hierarchy."""
    root = TaxonomyNode(name="root")

    # Create branch 1: root -> branch1 -> leaf1
    branch1 = TaxonomyNode(name="branch1")
    leaf1 = TaxonomyNode(name="leaf1")
    root.add_child("branch1", branch1)
    branch1.add_child("leaf1", leaf1)

    # Create branch 2: root -> branch2 -> subbranch -> leaf2
    branch2 = TaxonomyNode(name="branch2")
    subbranch = TaxonomyNode(name="subbranch")
    leaf2 = TaxonomyNode(name="leaf2")
    root.add_child("branch2", branch2)
    branch2.add_child("subbranch", subbranch)
    subbranch.add_child("leaf2", leaf2)

    # Test finding nodes in different branches
    assert root.find_path_to("leaf1") == ["root", "branch1", "leaf1"]
    assert root.find_path_to("leaf2") == ["root", "branch2", "subbranch", "leaf2"]
    assert root.find_path_to("subbranch") == ["root", "branch2", "subbranch"]
    assert root.find_path_to("branch1") == ["root", "branch1"]
    assert root.find_path_to("branch2") == ["root", "branch2"]


def test_find_path_to_from_non_root() -> None:
    """Test finding paths when starting from a non-root node."""
    root = TaxonomyNode(name="root")
    level1 = TaxonomyNode(name="level1")
    level2 = TaxonomyNode(name="level2")
    leaf = TaxonomyNode(name="leaf")

    root.add_child("level1", level1)
    level1.add_child("level2", level2)
    level2.add_child("leaf", leaf)

    # Start search from level1 instead of root
    path_from_level1 = level1.find_path_to("leaf")
    assert path_from_level1 == ["level1", "level2", "leaf"]

    # Should not find nodes that are ancestors or siblings
    assert level1.find_path_to("root") is None

    # Start search from level2
    path_from_level2 = level2.find_path_to("leaf")
    assert path_from_level2 == ["level2", "leaf"]


def test_find_path_to_with_evaluations() -> None:
    """Test that find_path_to works correctly with nodes that have evaluations."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])
    child = TaxonomyNode(name="child", checks=[mock_check_function_2], metrics=[mock_metric_function_2])
    parent.add_child("child", child)

    path = parent.find_path_to("child")

    # Path finding should work regardless of evaluations
    assert path == ["parent", "child"]

    # Original evaluations should be preserved
    assert parent.checks == [mock_check_function]
    assert child.checks == [mock_check_function_2]


def test_get_all_evaluations_single_node() -> None:
    """Test getting evaluations from a single node with no children."""
    node = TaxonomyNode(name="test", checks=[mock_check_function], metrics=[mock_metric_function])

    evaluations = node.get_all_evaluations()

    assert len(evaluations) == 2
    assert mock_metric_function in evaluations
    assert mock_check_function in evaluations


def test_get_all_evaluations_empty_node() -> None:
    """Test getting evaluations from a node with no evaluation functions."""
    node = TaxonomyNode(name="empty")

    evaluations = node.get_all_evaluations()

    assert evaluations == []


def test_get_all_evaluations_with_children() -> None:
    """Test getting evaluations from a node with children that have evaluations."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])
    child = TaxonomyNode(name="child", checks=[mock_check_function_2], metrics=[mock_metric_function_2])
    parent.add_child("child", child)

    evaluations = parent.get_all_evaluations()

    assert len(evaluations) == 4
    assert mock_check_function in evaluations
    assert mock_metric_function in evaluations
    assert mock_check_function_2 in evaluations
    assert mock_metric_function_2 in evaluations


def test_get_all_evaluations_deep_hierarchy() -> None:
    """Test getting evaluations from a deep hierarchy with evaluations at different levels."""
    root = TaxonomyNode(name="root", checks=[mock_check_function])
    level1 = TaxonomyNode(name="level1", metrics=[mock_metric_function])
    level2 = TaxonomyNode(name="level2")  # No evaluations
    leaf = TaxonomyNode(name="leaf", checks=[mock_check_function_2], metrics=[mock_metric_function_2])

    root.add_child("level1", level1)
    level1.add_child("level2", level2)
    level2.add_child("leaf", leaf)

    evaluations = root.get_all_evaluations()

    assert len(evaluations) == 4
    assert mock_check_function in evaluations
    assert mock_metric_function in evaluations
    assert mock_check_function_2 in evaluations
    assert mock_metric_function_2 in evaluations


def test_get_all_evaluations_deduplication() -> None:
    """Test that duplicate evaluation functions are only included once."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])
    child = TaxonomyNode(
        name="child",
        checks=[mock_check_function],  # Duplicate
        metrics=[mock_metric_function_2],
    )
    parent.add_child("child", child)

    evaluations = parent.get_all_evaluations()

    assert len(evaluations) == 3
    # Count occurrences manually since evaluations is a list
    check_count = sum(1 for func in evaluations if func is mock_check_function)
    assert check_count == 1, "Duplicate functions should only appear once"


def test_get_all_evaluations_multiple_children() -> None:
    """Test getting evaluations from a node with multiple children."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function])
    child1 = TaxonomyNode(name="child1", metrics=[mock_metric_function])
    child2 = TaxonomyNode(name="child2", checks=[mock_check_function_2])
    child3 = TaxonomyNode(name="child3", metrics=[mock_metric_function_2])

    parent.add_child("child1", child1)
    parent.add_child("child2", child2)
    parent.add_child("child3", child3)

    evaluations = parent.get_all_evaluations()

    assert len(evaluations) == 4
    assert mock_check_function in evaluations
    assert mock_metric_function in evaluations
    assert mock_check_function_2 in evaluations
    assert mock_metric_function_2 in evaluations


def test_get_all_evaluations_complex_deduplication() -> None:
    """Test deduplication in a complex hierarchy with multiple duplicates."""
    root = TaxonomyNode(
        name="root", checks=[mock_check_function, mock_check_function_2], metrics=[mock_metric_function]
    )

    # Branch 1 with duplicates
    branch1 = TaxonomyNode(
        name="branch1",
        checks=[mock_check_function],  # Duplicate
        metrics=[mock_metric_function_2],
    )
    leaf1 = TaxonomyNode(
        name="leaf1",
        checks=[mock_check_function_2],  # Duplicate
        metrics=[mock_metric_function],  # Duplicate
    )

    # Branch 2 with new and duplicate functions
    branch2 = TaxonomyNode(
        name="branch2",
        checks=[mock_check_function_3],  # New
        metrics=[mock_metric_function_3],  # New
    )

    root.add_child("branch1", branch1)
    branch1.add_child("leaf1", leaf1)
    root.add_child("branch2", branch2)

    evaluations = root.get_all_evaluations()

    # Should have 6 unique functions despite duplicates
    assert len(evaluations) == 6

    # Check that each function appears exactly once
    unique_functions = [
        mock_check_function,
        mock_check_function_2,
        mock_check_function_3,
        mock_metric_function,
        mock_metric_function_2,
        mock_metric_function_3,
    ]

    for func in unique_functions:
        count = sum(1 for f in evaluations if f is func)
        assert count == 1, f"Function {func.__name__} should appear exactly once"


def test_get_all_evaluations_order_preservation() -> None:
    """Test that evaluation order follows the traversal pattern."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])
    child = TaxonomyNode(name="child", checks=[mock_check_function_2], metrics=[mock_metric_function_2])
    parent.add_child("child", child)

    evaluations = parent.get_all_evaluations()

    # Parent's evaluations should come before child's evaluations
    # Within parent: metrics come before checks (based on implementation)
    parent_metric_index = evaluations.index(mock_metric_function)
    parent_check_index = evaluations.index(mock_check_function)
    child_metric_index = evaluations.index(mock_metric_function_2)
    child_check_index = evaluations.index(mock_check_function_2)

    # Parent evaluations should come before child evaluations
    assert parent_metric_index < child_metric_index
    assert parent_check_index < child_check_index


def test_get_all_evaluations_empty_children() -> None:
    """Test getting evaluations when children have no evaluation functions."""
    parent = TaxonomyNode(name="parent", checks=[mock_check_function], metrics=[mock_metric_function])
    empty_child1 = TaxonomyNode(name="empty_child1")  # No evaluations
    empty_child2 = TaxonomyNode(name="empty_child2")  # No evaluations

    parent.add_child("empty_child1", empty_child1)
    parent.add_child("empty_child2", empty_child2)

    evaluations = parent.get_all_evaluations()

    # Should only have parent's evaluations
    assert len(evaluations) == 2
    assert mock_check_function in evaluations
    assert mock_metric_function in evaluations
