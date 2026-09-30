from vitex.transcribe.aws_transcribe import AWSTranscriber


def _word(content, start, end):
    return {
        "type": "pronunciation",
        "start_time": str(start),
        "end_time": str(end),
        "alternatives": [{"content": content}],
    }


def _punct(content):
    return {"type": "punctuation", "alternatives": [{"content": content}]}


def test_groups_on_sentence_punctuation():
    payload = {
        "results": {
            "items": [
                _word("Hello", 0.0, 0.4),
                _word("world", 0.5, 0.9),
                _punct("."),
                _word("Next", 1.0, 1.3),
                _word("one", 1.4, 1.7),
                _punct("."),
            ]
        }
    }
    segs = AWSTranscriber._group_items(payload)
    assert len(segs) == 2
    assert segs[0].text == "Hello world."
    assert segs[0].start == 0.0 and segs[0].end == 0.9
    assert segs[1].text == "Next one."


def test_groups_on_long_pause():
    payload = {
        "results": {
            "items": [
                _word("A", 0.0, 0.2),
                _word("B", 5.0, 5.2),  # >1.2s gap -> new segment
            ]
        }
    }
    segs = AWSTranscriber._group_items(payload)
    assert len(segs) == 2
