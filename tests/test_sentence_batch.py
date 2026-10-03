"""SentenceBatchAggregator: lead/carry rhythm, zero text loss, lifecycle."""

from services.sentence_batch import SentenceBatchAggregator

FOUR = (
    "One sentence here. Second one follows! Third arrives last. Fourth closes it out."
)
THREE = (
    "Did you know that honey never spoils?"
    " Archaeologists found pots of honey in Egyptian tombs. It is remarkable."
)


async def _feed(agg: SentenceBatchAggregator, stream: str, chunk: int) -> list[str]:
    """Feed stream in fixed-size chunks; return released texts incl. flush tail."""
    got: list[str] = []
    for i in range(0, len(stream), chunk):
        async for out in agg.aggregate(stream[i : i + chunk]):
            got.append(out.text)
    tail = await agg.flush()
    if tail:
        got.append(tail.text)
    return got


async def test_lead2_carry2_shape_matches_ab_winner() -> None:
    """lead=2/carry=2 (the A/B winner config: no gaps, TTFA 1.5-2.5s)."""
    agg = SentenceBatchAggregator(lead=2, carry=2)
    got = await _feed(agg, FOUR, 5)
    assert len(got) == 2
    assert got[0] == "One sentence here. Second one follows!"
    assert got[1] == "Third arrives last. Fourth closes it out."


async def test_lead1_carry2_single_first_pair_after() -> None:
    """lead=1/carry=2: first release is sentence 1 alone, then pairs."""
    agg = SentenceBatchAggregator(lead=1, carry=2)
    got = await _feed(agg, FOUR, 5)
    assert got[0] == "One sentence here."
    assert got[1] == "Second one follows! Third arrives last."
    assert got[2] == "Fourth closes it out."
    assert sorted(" ".join(got).split()) == sorted(FOUR.split())


async def test_zero_loss_across_chunk_sizes() -> None:
    """Every original character accounted for at token-ish and large chunks."""
    for chunk in (1, 3, 5, 40, 200):
        agg = SentenceBatchAggregator(lead=1, carry=2)
        got = await _feed(agg, FOUR, chunk)
        assert sum(len(g) for g in got) <= len(FOUR), f"chunk {chunk}"
        words_out = sorted(" ".join(got).split())
        assert words_out == sorted(FOUR.split())


async def test_no_duplicate_words_across_releases() -> None:
    """Text appears exactly once across releases (no double-spoken audio)."""
    agg = SentenceBatchAggregator(lead=2, carry=2)
    got = await _feed(agg, FOUR, 3)
    out_words = " ".join(got).split()
    assert sorted(out_words) == sorted(FOUR.split())


async def test_flush_releases_odd_tail() -> None:
    """A stream ending with an unfilled quota releases the tail via flush."""
    agg = SentenceBatchAggregator(lead=2, carry=2)
    got = await _feed(agg, THREE, 6)
    # THREE sentences: pair1 released mid-stream, tail = sentence 3
    assert got[0] == ("Did you know that honey never spoils?"
                      " Archaeologists found pots of honey in Egyptian tombs.")
    assert got[1] == "It is remarkable."
    assert sorted(" ".join(got).split()) == sorted(THREE.split())


async def test_quota_resets_to_lead_after_flush() -> None:
    """flush() restores the lead quota for the next turn."""
    agg = SentenceBatchAggregator(lead=1, carry=2)
    await _feed(agg, THREE, 6)
    assert agg._quota == 1
    again = await _feed(agg, "New opener here. Now the second sentence. Third one.", 5)
    assert again[0] == "New opener here."


async def test_interruption_discards_and_resets_quota() -> None:
    """handle_interruption clears the buffer and restores the lead quota."""
    agg = SentenceBatchAggregator(lead=1, carry=2)
    async for _ in agg.aggregate("Sentence number one. Pendin"):
        pass
    await agg.handle_interruption()
    tail = await agg.flush()
    assert tail is None
    assert agg._quota == 1


async def test_reset_clears_buffer_and_quota() -> None:
    """reset() discards partial text and restores the lead quota."""
    agg = SentenceBatchAggregator(lead=1, carry=2)
    async for _ in agg.aggregate("Half a senten"):
        pass
    await agg.reset()
    tail = await agg.flush()
    assert tail is None
    assert agg._quota == 1


async def test_empty_and_whitespace_streams_produce_nothing() -> None:
    """Empty and whitespace-only streams never produce releases."""
    agg = SentenceBatchAggregator(lead=1, carry=2)
    got = await _feed(agg, "", 4)
    got += await _feed(agg, "   ", 4)
    assert got == []
