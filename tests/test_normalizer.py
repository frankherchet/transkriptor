from __future__ import annotations

from transkriptor.normalizer import extract_raw_segments, normalize_segments


def test_extract_raw_segments_from_dict() -> None:
    assert extract_raw_segments({"segments": [{"text": "hi"}]}) == [{"text": "hi"}]


def test_normalize_segments_offsets_and_namespaces_speakers() -> None:
    result = normalize_segments(
        [
            {
                "speaker": "SPEAKER_00",
                "start": 1.2345,
                "end": 2.0,
                "text": " hello ",
            }
        ],
        chunk_index=2,
        offset_seconds=100,
        first_id=3,
    )

    assert result == [
        {
            "id": 3,
            "speaker": "chunk2:SPEAKER_00",
            "speaker_local": "SPEAKER_00",
            "chunk_index": 2,
            "start": 101.234,
            "end": 102.0,
            "text": "hello",
        }
    ]
