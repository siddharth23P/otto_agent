

# ---- temperature clamping -------------------------------------------------


def test_a_temperature_below_the_documented_minimum_is_clamped_not_sent_raw():
    """Inception resets an out-of-range temperature to the model default (1.0
    for mercury-2.5) rather than clamping it, so asking for 0.0 was being
    served at maximum randomness -- for the evaluator, whose whole job is a
    verdict that should not be a dice roll."""
    from agent.router.llm_provider.inception_provider import _clamp_temperature

    assert _clamp_temperature(0.0) == 0.5
    assert _clamp_temperature(0.2) == 0.5
    assert _clamp_temperature(0.7) == 0.7
    assert _clamp_temperature(1.5) == 1.0
    assert _clamp_temperature(None) is None
