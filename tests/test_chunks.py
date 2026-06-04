from __future__ import annotations

import pytest

from transkriptor.chunks import build_chunk_specs, parse_chunk_markers, parse_time_marker


def test_parse_time_marker_units_and_clock_values() -> None:
    assert parse_time_marker("100s") == 100
    assert parse_time_marker("23m") == 1380
    assert parse_time_marker("1h") == 3600
    assert parse_time_marker("01:02") == 62
    assert parse_time_marker("01:02:03") == 3723


def test_parse_chunk_markers() -> None:
    assert parse_chunk_markers("100s,23m,59m") == [100, 1380, 3540]


@pytest.mark.parametrize("value", ["0s", "23m,100s", "10s,10s", "abc"])
def test_parse_chunk_markers_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_chunk_markers(value)


def test_build_chunk_specs() -> None:
    specs = build_chunk_specs([100, 200])
    assert [(spec.index, spec.start, spec.end) for spec in specs] == [
        (0, 0.0, 100),
        (1, 100, 200),
        (2, 200, None),
    ]
