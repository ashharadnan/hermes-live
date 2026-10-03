"""TwoSentenceAggregator: pairing rhythm, zero text loss, lifecycle semantics."""

from services.two_sentence import TwoSentenceAggregator

FOUR = (
    "One sentence here. Second one follows! Third arrives last. Fourth closes it out."
)
EXPECTED_PAIRS = [
    "One sentence here. Second one follows!",
    "Third arrives last. Fourth closes it out.",
]


async def _feed(agg: TwoSentenceAggregator, stream: str, chunk: int) -> list[str]:
    """Feed stream in fixed-size chunks; return released texts incl. flush tail."""
    got: list[str] = []
    for i in range(0, len(stream), chunk):
        async for out in agg.aggregate(stream[i : i + chunk]):
            got.append(out.text)
    tail = await agg.flush()
    if tail:
        got.append(tail.text)
    return got


async def test_releases_pairs_and_tail_without_loss() -> None:
    """4 sentences -> 2 releases (pair, tail); every original char accounted for."""
    for chunk in (1, 5, 40, 200):  # token-ish, small, large, whole-string
        got = await _feed(TwoSentenceAggregator(), FOUR, chunk)
        assert len(got) == 2, f"chunk {chunk}: got {got}"
        assert got[0] == EXPECTED_PAIRS[0]
        assert got[1] == EXPECTED_PAIRS[1]
        assert sum(len(g) for g in got) + 1 == len(FOUR)  # +1 = join space


async def test_even_boundary_flush_releases_last_pair() -> None:
    """A stream ending exactly on a pair boundary flushes nothing extra."""
    agg = TwoSentenceAggregator()
    got = await _feed(agg, "Alpha starts here. Beta ends it!", 7)
    assert got == ["Alpha starts here. Beta ends it!"]
    assert agg._buf == ""


async def test_flush_after_odd_sentence_releases_tail() -> None:
    """Odd count streams release the unpaired sentence through flush."""
    agg = TwoSentenceAggregator()
    got = await _feed(agg, "First one alone. Second one too.", 6)
    assert got == ["First one alone. Second one too."]


async def test_empty_and_whitespace_streams_produce_nothing() -> None:
    """Empty and whitespace-only streams never produce releases."""
    agg = TwoSentenceAggregator()
    got = await _feed(agg, "", 4)
    got += await _feed(agg, "   ", 4)
    assert got == []


async def test_handle_interruption_discards_pending_text() -> None:
    """An unpaired pending sentence dies on interruption; next turn starts clean."""
    agg = TwoSentenceAggregator()
    got = []
    async for out in agg.aggregate("Sentence number one. Pendin"):
        got.append(out.text)
    await agg.handle_interruption()
    tail = await agg.flush()
    assert tail is None
    assert got == []


async def test_reset_clears_buffer() -> None:
    """reset() discards half-accumulated text so flush yields nothing."""
    agg = TwoSentenceAggregator()
    async for _ in agg.aggregate("Half a senten"):
        pass
    await agg.reset()
    tail = await agg.flush()
    assert tail is None


async def test_pending_pair_never_crosses_release_boundary() -> None:
    """No sentence text appears twice across releases (would duplicate audio)."""
    agg = TwoSentenceAggregator()
    got = await _feed(agg, FOUR, 3)
    words = FOUR.split()
    out_words = " ".join(got).split()
    assert len(out_words) == len(words)
    assert sorted(out_words) == sorted(words)
