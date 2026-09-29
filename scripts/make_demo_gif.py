"""Generate a tiny deterministic GIF walkthrough for the README."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"
W, H = 640, 400
COLORS = {
    "bg": "#f4f5f0", "white": "#ffffff", "ink": "#18302e",
    "muted": "#63746f", "green": "#1e5e4b", "border": "#d9dfd7",
    "accent": "#49685e",
}


def font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
             else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
             else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def rounded(draw, box, radius=8, fill=None, outline=None):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def base_frame():
    c = COLORS
    image = Image.new("RGB", (W, H), c["bg"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 38), fill=c["white"])
    draw.line((0, 38, W, 38), fill=c["border"])
    draw.text((25, 13), "EXPRESS DOCS · RESEARCH DEMO", font=font(8, True), fill=c["ink"])
    draw.text((30, 61), "SELF-CORRECTIVE RAG", font=font(8, True), fill=c["accent"])
    draw.text((30, 78), "Ask the docs.", font=font(25, True), fill=c["ink"])
    draw.text((30, 107), "Check the source.", font=font(25, True), fill=c["ink"])
    draw.text((30, 143), "Grounded answers with citations and verification.", font=font(10), fill=c["muted"])

    rounded(draw, (30, 169, 400, 368), 10, c["white"], c["border"])
    draw.text((46, 185), "Your question", font=font(9, True), fill=c["ink"])
    rounded(draw, (46, 205, 384, 246), 7, c["white"], "#bdccc4")
    draw.text((58, 220), "How does FastAPI validate an integer path parameter?", font=font(9), fill=c["ink"])
    rounded(draw, (46, 258, 116, 285), 6, c["green"], c["green"])
    draw.text((61, 267), "Ask", font=font(9, True), fill=c["white"])

    rounded(draw, (419, 60, 612, 170), 10, c["white"], c["border"])
    draw.text((434, 75), "EVIDENCE", font=font(8, True), fill=c["accent"])
    draw.text((434, 94), "Sources", font=font(13, True), fill=c["ink"])

    rounded(draw, (419, 184, 612, 368), 10, c["white"], c["border"])
    draw.text((434, 199), "CORPUS", font=font(8, True), fill=c["accent"])
    draw.text((434, 218), "Indexed documents", font=font(13, True), fill=c["ink"])
    for i, label in enumerate(("Path parameters", "Request body", "Dependencies", "Response model")):
        draw.text((434, 247 + i * 25), label, font=font(8), fill=c["ink"])
    return image


def main():
    c = COLORS
    frames = []

    image = base_frame()
    draw = ImageDraw.Draw(image)
    draw.text((434, 125), "Sources appear after", font=font(9), fill=c["muted"])
    draw.text((434, 140), "an answer.", font=font(9), fill=c["muted"])
    draw.text((135, 266), "Ready", font=font(8), fill=c["muted"])
    frames.append(image)

    image = base_frame()
    draw = ImageDraw.Draw(image)
    draw.text((46, 303), "FastAPI parses the URL value using the int annotation [S1].", font=font(9), fill=c["ink"])
    draw.text((46, 321), "Invalid values return a validation error before the route runs.", font=font(9), fill=c["ink"])
    draw.text((135, 266), "1 local attempt · verified", font=font(8), fill=c["muted"])
    draw.text((434, 126), "[S1] FastAPI path parameters", font=font(8), fill=c["ink"])
    draw.text((434, 143), "fastapi.tiangolo.com/…", font=font(8), fill="#1a6b50")
    frames.append(image)

    image = base_frame()
    draw = ImageDraw.Draw(image)
    rounded(draw, (210, 150, 596, 333), 9, "#fafcf9", "#97b4a5")
    draw.text((226, 165), "HOW THIS ANSWER WAS CHECKED", font=font(8, True), fill=c["accent"])
    for i, step in enumerate((
        "✓ Query analyzed",
        "✓ Local retrieval · 1 attempt",
        "✓ Relevant chunks graded",
        "✓ 1 source cited",
        "✓ Support verification passed",
    )):
        draw.text((228, 193 + i * 25), step, font=font(10), fill=c["ink"])
    frames.append(image)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT, save_all=True, append_images=frames[1:],
        duration=[1100, 1400, 1800], loop=0, optimize=True, disposal=2,
    )
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

# This generator is documentation-only and is not required at runtime.
