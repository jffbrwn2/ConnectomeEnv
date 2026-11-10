"""Lossless wire encoding for image-heavy OpenRouter conversations."""
import base64
from functools import lru_cache
from io import BytesIO
from PIL import Image

@lru_cache(maxsize=64)
def lossless_data_url(png_url):
    """Keep every pixel and image position; use WebP only when smaller."""
    prefix = "data:image/png;base64,"
    if not png_url.startswith(prefix):
        return png_url
    data = base64.b64decode(png_url[len(prefix):], validate=True)
    with Image.open(BytesIO(data)) as image:
        if image.mode not in ("RGB", "RGBA") or getattr(image, "is_animated", False):
            return png_url
        output = BytesIO()
        image.save(output, format="WEBP", lossless=True, exact=True, method=1)
    encoded = output.getvalue()
    if len(encoded) >= len(data):
        return png_url
    return "data:image/webp;base64," + base64.b64encode(encoded).decode()


def deduplicate_images(payload):
    """Replace exact repeated inline images with references; retain all unique views.

    Returns a new payload. First occurrences are labelled so later references are
    unambiguous without changing image order or any tool-call/result messages.
    """
    seen = {}
    messages = []
    for message in payload.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            messages.append(message)
            continue
        blocks = []
        for block in content:
            url = block.get('image_url', {}).get('url', '') if block.get('type') == 'image_url' else ''
            if not url.startswith('data:image/'):
                blocks.append(block)
                continue
            if url in seen:
                blocks.append({'type': 'text', 'text': f"Image identical to {seen[url]} shown earlier in this conversation (every pixel unchanged)."})
            else:
                label = f'image-{len(seen) + 1}'
                seen[url] = label
                blocks.extend([{'type': 'text', 'text': f'Image reference: {label}'}, block])
        messages.append({**message, 'content': blocks})
    return {**payload, 'messages': messages}


def inline_image_bytes(payload):
    return sum(len(block.get('image_url', {}).get('url', ''))
               for message in payload.get('messages', [])
               if isinstance(message.get('content'), list)
               for block in message['content'] if block.get('type') == 'image_url')
