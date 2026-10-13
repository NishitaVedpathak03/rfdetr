"""Model-independent hierarchy checks and foreground mask."""
import numpy as np


def validate_hierarchy(nodes):
    if not nodes:
        return
    lookup = {n["id"]: n for n in nodes}
    if len(lookup) != len(nodes) or sum(n["parent_id"] is None for n in nodes) != 1:
        raise ValueError("Hierarchy requires unique IDs and exactly one root")
    for node in nodes:
        if len(set(node["children"])) != len(node["children"]):
            raise ValueError("Duplicate child reference")
        if node["parent_id"] is not None:
            if node["parent_id"] not in lookup or node["id"] not in lookup[node["parent_id"]]["children"]:
                raise ValueError("Parent and child references disagree")
        seen, cur = set(), node["id"]
        while cur is not None:
            if cur not in lookup or cur in seen:
                raise ValueError("Invalid parent reference or cycle")
            seen.add(cur)
            cur = lookup[cur]["parent_id"]
        for child in node["children"]:
            if child not in lookup or lookup[child]["parent_id"] != node["id"]:
                raise ValueError("Inconsistent child reference")


def ink_mask(image):
    """Dark foreground relative to border background; no colored-image dependency."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return gray < min(220, float(np.median(border)) - 25)
