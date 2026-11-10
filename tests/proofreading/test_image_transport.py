import base64
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from proofreading.image_transport import lossless_data_url

@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_lossless_wire_image_preserves_pixels_and_dimensions(mode):
    gray = np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8)
    pixels = np.repeat(gray[:, :, None], len(mode), axis=2)
    if mode == "RGBA":
        pixels[::2, :, 3] = 0
    buf = BytesIO()
    Image.fromarray(pixels, mode).save(buf, format="PNG", compress_level=0)
    original = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    converted = lossless_data_url(original)
    assert converted.startswith("data:image/webp;base64,")
    decoded = Image.open(BytesIO(base64.b64decode(converted.split(",", 1)[1])))
    np.testing.assert_array_equal(np.asarray(decoded.convert(mode)), pixels)
    assert len(converted) < len(original)
    assert lossless_data_url(original) == converted

def test_preserves_non_png_and_high_bit_depth_images():
    assert lossless_data_url("https://example.org/view.png") == "https://example.org/view.png"
    buf = BytesIO()
    Image.fromarray(np.arange(256, dtype=np.uint16).reshape(16, 16)).save(buf, format="PNG")
    original = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert lossless_data_url(original) == original


def test_repeated_images_keep_all_unique_evidence_and_tool_messages():
    import copy
    from proofreading.image_transport import deduplicate_images
    def image(value):
        return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + value, 'detail': 'high'}}
    a, b = image('AAAA'), image('BBBB')
    payload = {'model': 'test', 'messages': [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Inspect.'}, a]},
        {'role': 'assistant', 'tool_calls': [{'id': 'call-1'}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': '{"ok":true}'},
        {'role': 'user', 'content': [a, b, a]}]}
    original = copy.deepcopy(payload)
    result = deduplicate_images(payload)
    assert payload == original
    images = [block for message in result['messages'] if isinstance(message.get('content'), list)
              for block in message['content'] if block['type'] == 'image_url']
    assert images == [a, b]
    assert result['messages'][1:3] == original['messages'][1:3]
    assert result['messages'][3]['content'][0]['text'].startswith('Image identical to image-1')
    assert result['messages'][3]['content'][-1]['text'].startswith('Image identical to image-1')
