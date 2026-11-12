from thesis_assistant.vision import format_image_snippets, guess_image_mime


def test_guess_image_mime_detects_common_formats():
    assert guess_image_mime(b"\x89PNG\r\n\x1a\n" + b"0" * 8) == "image/png"
    assert guess_image_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert guess_image_mime(b"GIF89a....") == "image/gif"
    assert guess_image_mime(b"RIFF\x00\x00\x00\x00WEBP") == "image/webp"


def test_guess_image_mime_defaults_to_png():
    assert guess_image_mime(b"unknown") == "image/png"


def test_format_image_snippets_numbers_images():
    block = format_image_snippets(["first", "second"])
    assert "[Embedded image 1]\nfirst" in block
    assert "[Embedded image 2]\nsecond" in block
