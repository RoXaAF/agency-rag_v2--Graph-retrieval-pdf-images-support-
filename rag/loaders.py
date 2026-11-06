"""txt / pdf / image files -> chunk dicts {id, source, page, type, image_path, text}."""
import time

import pymupdf

import config
from rag import llm

CAPTION_PROMPT = (
    "Describe this image in detail. Transcribe all visible text exactly. "
    "For charts and tables, state every value you can read (labels and numbers)."
)


def split(text, size=None, overlap=None):
    size, overlap = size or config.CHUNK_SIZE, overlap or config.CHUNK_OVERLAP
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # prefer to cut at a space
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def caption(image_bytes):
    """Describe an image with the vision model. Cached by image hash."""
    k = llm.key(image_bytes, CAPTION_PROMPT, config.LLM_MODEL)
    t = time.time()
    import base64
    text, hit = llm.cached("captions", k, lambda: llm.chat([{
        "role": "user", "content": CAPTION_PROMPT, "images": [base64.b64encode(image_bytes).decode()]}]))
    print(f"    caption {'(cached)' if hit else f'{time.time() - t:.1f}s'}")
    return text.strip()


def _image_chunk(cid, source, page, path, data, w, h):
    if min(w, h) < config.MIN_IMAGE_PX:
        print(f"    skip small image {cid} ({w}x{h})")
        return None
    return {"id": cid, "source": source, "page": page, "type": "image",
            "image_path": str(path), "text": caption(data)}


def load_txt(path):
    return [{"id": f"{path.name}#{i}", "source": path.name, "page": 0, "type": "text", "image_path": None, "text": t}
            for i, t in enumerate(split(path.read_text(encoding="utf-8")), 1)]


def load_pdf(path):
    chunks = []
    doc = pymupdf.open(path)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for pno, page in enumerate(doc, 1):
        for i, t in enumerate(split(page.get_text()), 1):
            chunks.append({"id": f"{path.name}#p{pno}#{i}", "source": path.name, "page": pno,
                           "type": "pdf_text", "image_path": None, "text": t})
        for n, img in enumerate(page.get_images(full=True), 1):
            info = doc.extract_image(img[0])
            out = config.IMAGES_DIR / f"{path.stem}_p{pno}_{n}.{info['ext']}"
            out.write_bytes(info["image"])
            c = _image_chunk(f"{path.name}#p{pno}#img{n}", path.name, pno, out.relative_to(config.ROOT),
                             info["image"], info["width"], info["height"])
            if c:
                chunks.append(c)
    return chunks


def load_image(path):
    data = path.read_bytes()
    pix = pymupdf.Pixmap(data)
    c = _image_chunk(f"{path.name}#img", path.name, 0, path.relative_to(config.ROOT), data, pix.width, pix.height)
    return [c] if c else []


LOADERS = {".txt": load_txt, ".pdf": load_pdf, ".png": load_image, ".jpg": load_image, ".jpeg": load_image}
