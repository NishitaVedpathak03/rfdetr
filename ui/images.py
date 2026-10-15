"""Equation image decoding and active-region overlay."""
import io
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_COLORS = {"base": "#2878D0", "superscript": "#24A46B", "numerator": "#9348BD",
                  "denominator": "#159EAD", "operator": "#E53935", "leaf": "#E5A000",
                  "root": "#DB60B0", "function": "#8661C1", "group": "#666666"}

BOX_PADDING = 8  # Applied after resizing, so the visible outline clears small glyphs.


def render_overlay(image: Image.Image, nodes: list[dict], colors: dict, *, show_boxes=True,
                   show_labels=False, show_confidence=False, selected_id=None,
                   active_id=None, child_ids=(), display_size=None) -> Image.Image:
    canvas = image.convert("RGB").copy()
    if display_size:
        canvas.thumbnail(display_size, Image.Resampling.LANCZOS)
    sx, sy = canvas.width / image.width, canvas.height / image.height
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)

    def scaled_box(node):
        x1, y1, x2, y2 = node["bbox"]
        # Source boxes use exclusive right/bottom; Pillow rectangles are inclusive.
        left = max(0, min(canvas.width - 1, x1 * sx - BOX_PADDING))
        top = max(0, min(canvas.height - 1, y1 * sy - BOX_PADDING))
        right = max(left, min(canvas.width - 1, x2 * sx - 1 + BOX_PADDING))
        bottom = max(top, min(canvas.height - 1, y2 * sy - 1 + BOX_PADDING))
        return left, top, right, bottom

    for node in sorted(nodes, key=lambda n: n.get("depth", 0)):
        box = scaled_box(node)
        color = colors.get(node["role"], "#64748B")
        if show_boxes and node["children"]:
            draw.rectangle(box, outline=color, width=1)
        if show_labels:
            label = f"#{node['id']} {node['kind']}"
            confidence = node.get("confidence")
            if show_confidence and isinstance(confidence, (int, float)) and math.isfinite(confidence):
                label += f" {confidence:.2f}"
            bounds = draw.textbbox((0, 0), label, font=font)
            label_width = bounds[2] - bounds[0]
            x = max(0, min(box[0], canvas.width - label_width - 4))
            y = max(0, min(box[1] - 15, canvas.height - 16))
            draw.rectangle((x, y, min(canvas.width - 1, x + label_width + 4), y + 15), fill="white")
            draw.text((x + 2, y), label, fill=color, font=font)
    lookup = {node["id"]: node for node in nodes}
    for child in child_ids:
        if child in lookup:
            draw.rectangle(scaled_box(lookup[child]), outline="#93C5FD", width=2)
    for node_id, color, width in ((active_id, "#2563EB", 1), (selected_id, "#1D4ED8", 1)):
        if node_id in lookup:
            draw.rectangle(scaled_box(lookup[node_id]), outline=color, width=width)
    return canvas


def uploaded_image(data: bytes) -> Image.Image:
    """Decode pixels only; a preview is never assigned demo annotations."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("Please upload a PNG or JPEG image.")
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")
