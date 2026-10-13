"""Validate and describe RF-DETR spatial hierarchy nodes."""
import math


def normalize_nodes(raw_nodes: list[dict], image_size: tuple[int, int]) -> list[dict]:
    """Validate original-coordinate boxes and derive children from parent IDs."""
    if not isinstance(raw_nodes, list):
        raise ValueError("Annotations must be a list of nodes.")
    nodes, lookup = [], {}
    width, height = image_size
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise ValueError("Each annotation must be an object.")
        node = dict(raw)
        node_id = node.get("id")
        if type(node_id) is not int or node_id in lookup:
            raise ValueError("Node IDs must be unique integers.")
        parent = node.get("parent_id")
        if "parent_id" not in node or (parent is not None and type(parent) is not int):
            raise ValueError(f"Node {node_id}: invalid parent ID.")
        if not all(isinstance(node.get(key), str) for key in ("kind", "role")):
            raise ValueError(f"Node {node_id}: kind and role must be strings.")
        box = node.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not all(
            type(value) in (int, float) and math.isfinite(value) for value in box
        ):
            raise ValueError(f"Node {node_id}: invalid bounding box.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Node {node_id}: box lies outside the original image or has no area.")
        node["children"] = []
        nodes.append(node)
        lookup[node_id] = node
    roots = []
    for node in nodes:
        parent = node["parent_id"]
        if parent is None:
            roots.append(node["id"])
        elif parent not in lookup or parent == node["id"]:
            raise ValueError(f"Node {node['id']}: missing parent or self-reference.")
        else:
            lookup[parent]["children"].append(node["id"])
    if nodes and len(roots) != 1:
        raise ValueError("The annotation hierarchy must have exactly one root.")
    visited, stack = set(), [(node_id, 0) for node_id in roots]
    while stack:
        node_id, depth = stack.pop()
        if node_id in visited:
            raise ValueError("The annotation hierarchy contains a cycle.")
        visited.add(node_id)
        lookup[node_id]["depth"] = depth
        stack.extend((child, depth + 1) for child in lookup[node_id]["children"])
    if len(visited) != len(nodes):
        raise ValueError("Some nodes are disconnected or belong to a cycle.")
    for raw, node in zip(raw_nodes, nodes):
        if "children" in raw:
            if not isinstance(raw["children"], list) or sorted(raw["children"]) != sorted(node["children"]):
                raise ValueError(f"Node {node['id']}: children disagree with parent IDs.")
            node["children"] = list(raw["children"])
    return nodes


def validate_steps(raw_steps, nodes: list[dict]) -> list[dict]:
    """Preserve recorded order. Never generate replacement demo steps."""
    steps = raw_steps.get("steps") if isinstance(raw_steps, dict) else raw_steps
    if not isinstance(steps, list):
        raise ValueError("steps.json must contain a list of steps.")
    lookup = {node["id"]: node for node in nodes}
    for position, step in enumerate(steps):
        if not isinstance(step, dict) or type(step.get("node_id")) is not int or step["node_id"] not in lookup:
            raise ValueError(f"Step {position + 1} references an unknown node.")
        node = lookup[step["node_id"]]
        if step.get("parent_id") != node["parent_id"] or step.get("child_ids") != node["children"]:
            raise ValueError(f"Step {position + 1}: parent or children disagree with annotations.")
        if step.get("bbox") != node["bbox"] or step.get("depth") != node["depth"]:
            raise ValueError(f"Step {position + 1}: region or depth disagree with annotations.")
        if any(step.get(key) != node.get(key) for key in ("kind", "role", "text")):
            raise ValueError(f"Step {position + 1}: description disagrees with annotations.")
        expected = "divide_region" if node["children"] else "mark_leaf"
        if step.get("action") != expected:
            raise ValueError(f"Step {position + 1}: action disagrees with the node's children.")
    return steps


def step_description(step: dict, node: dict, predicted=False) -> str:
    if step["action"] == "mark_leaf":
        text = f" '{node['text']}'" if node.get("text") is not None else ""
        return f"Mark terminal {node['kind']}{text}; this region has no child nodes."
    traversal = 'predicted hierarchy' if predicted else 'recorded traversal'
    return f"Divide the {node['kind']} region into {len(node['children'])} child regions, following the {traversal}."
