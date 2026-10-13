"""Foreground-only coloring, hierarchy overlays, and animation-ready steps."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy import ndimage

from .hierarchy import ink_mask, validate_hierarchy


ROLE_COLORS = {"base": "#2878D0", "superscript": "#24A46B", "numerator": "#9348BD",
               "denominator": "#159EAD", "operator": "#E53935", "leaf": "#E5A000",
               "root": "#DB60B0", "function": "#8661C1", "group": "#666666"}


def recursive_steps(nodes):
    validate_hierarchy(nodes)
    if not nodes:
        return []
    lookup = {n["id"]: n for n in nodes}
    root = next(n for n in nodes if n["parent_id"] is None)
    steps, stack = [], [(root["id"], 0)]
    while stack:
        node_id, depth = stack.pop()
        node = lookup[node_id]
        steps.append({"step": len(steps), "node_id": node_id, "parent_id": node["parent_id"],
                      "depth": depth, "action": "divide_region" if node["children"] else "mark_leaf",
                      "kind": node["kind"], "role": node["role"], "text": node.get("text"),
                      "bbox": node["bbox"], "child_ids": node["children"],
                      "confidence": node.get("confidence"), "color": ROLE_COLORS.get(node["role"], "#666666")})
        stack.extend((child, depth + 1) for child in reversed(node["children"]))
    return steps


def contextual_role(node, lookup):
    if node["role"] != "group":
        return node["role"]
    parent = node["parent_id"]
    while parent is not None:
        ancestor = lookup[parent]
        if ancestor["role"] in {"base", "superscript", "numerator", "denominator"}:
            return ancestor["role"]
        parent = ancestor["parent_id"]
    return "group"


def colored_equation(image, nodes):
    """Assign each connected ink component to at most one terminal region.

    This is a heuristic pixel mask, not a supervised instance segmentation head.
    Ambiguous components remain untouched instead of bleeding into nearby glyphs.
    """
    rgb = np.asarray(image.convert("RGB")).copy()
    foreground = ink_mask(image)
    components, count = ndimage.label(foreground)
    slices = ndimage.find_objects(components)
    terminals = [n for n in nodes if n.get("terminal", n["kind"] in {"leaf", "symbol"})]
    lookup = {n["id"]: n for n in nodes}
    colored, ambiguous = 0, 0
    for component_id, bounds in enumerate(slices, 1):
        if bounds is None:
            continue
        ys, xs = bounds
        yy, xx = np.where(components[bounds] == component_id)
        yy, xx = yy + ys.start, xx + xs.start
        candidates = []
        for node in terminals:
            x1, y1, x2, y2 = node["bbox"]
            inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
            coverage = float(inside.mean())
            if coverage < .85:
                continue
            component_area = (xs.stop - xs.start) * (ys.stop - ys.start)
            box_area = max(1, (x2 - x1) * (y2 - y1))
            score = coverage + .2 * min(1, component_area / box_area) + .1 * (node.get("confidence") or 0)
            candidates.append((score, node))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .025:
            ambiguous += 1
            continue
        node = candidates[0][1]
        color = np.array(ImageColor.getrgb(ROLE_COLORS.get(contextual_role(node, lookup), "#666666")))
        # Preserve antialiasing/shading while changing only dark foreground pixels.
        darkness = 1 - rgb[yy, xx].mean(1) / 255.
        rgb[yy, xx] = np.uint8(255 * (1 - darkness[:, None]) + color * darkness[:, None])
        colored += 1
    return Image.fromarray(rgb), {"ink_components": count, "colored_components": colored,
                                  "ambiguous_components_left_unchanged": ambiguous}


def hierarchy_overlay(image, nodes):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    lookup = {n["id"]: n for n in nodes}
    # Edges first; draw boxes and labels on top.
    for node in nodes:
        if node["parent_id"] is None:
            continue
        parent = lookup[node["parent_id"]]
        a, b = node["bbox"], parent["bbox"]
        draw.line(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2,
                   (b[0] + b[2]) / 2, (b[1] + b[3]) / 2), fill="#B0B0B0", width=1)
    for node in sorted(nodes, key=lambda n: (n["bbox"][2] - n["bbox"][0]) * (n["bbox"][3] - n["bbox"][1]), reverse=True):
        color = ROLE_COLORS.get(node["role"], "#666666")
        box = node["bbox"]
        draw.rectangle(box, outline=color, width=1 if node.get("terminal") else 2)
        text = f"{node['id']} {node['kind']}:{node['role']}"
        # ASCII labels work with Pillow's bundled font on Windows and Kaggle.
        text_box = draw.textbbox((box[0], max(0, box[1] - 12)), text)
        draw.rectangle(text_box, fill="white")
        draw.text((box[0], max(0, box[1] - 12)), text, fill=color)
    return canvas


def save_visualizations(image, nodes, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    colored, stats = colored_equation(image, nodes)
    colored.save(out / "predicted_colored.png")
    hierarchy_overlay(image, nodes).save(out / "predicted_regions.png")
    return stats
