import pytest

from examples.experimental.rafo.credit import EventOutcome, event_weights, fixed_coefficients, rloo


class TestRloo:
    def test_each_reward_is_compared_with_the_mean_of_the_others(self) -> None:
        assert rloo([1.0, 0.0, 0.0, 0.0]) == pytest.approx([1.0, -1 / 3, -1 / 3, -1 / 3])

    def test_a_group_with_one_outcome_has_no_signal(self) -> None:
        assert rloo([0.0, 0.0]) == [0.0, 0.0]
        assert rloo([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]

    def test_a_group_needs_two_samples(self) -> None:
        with pytest.raises(ValueError, match="2"):
            rloo([1.0])


class TestEventWeights:
    def test_own_credit_is_the_mean_reward_of_the_event_continuations(self) -> None:
        events = [EventOutcome(parent=None, parent_cont=None, cont_rewards=(1.0, 0.0))]
        assert event_weights(events, credit="own") == [0.5]

    def test_subtree_credit_gives_the_first_turn_the_weight_of_the_review_that_rescued_its_branch(self) -> None:
        """Event 1 reviews again inside continuation 0 of event 0; its success also credits event 0's turn."""
        events = [
            EventOutcome(parent=None, parent_cont=None, cont_rewards=(0.0, 0.0)),
            EventOutcome(parent=0, parent_cont=0, cont_rewards=(1.0, 0.0)),
        ]

        assert event_weights(events, credit="own") == [0.0, 0.5]
        assert event_weights(events, credit="subtree") == [0.25, 0.5]

    def test_subtree_credit_is_computed_from_the_deepest_event_up(self) -> None:
        events = [
            EventOutcome(parent=None, parent_cont=None, cont_rewards=(0.0, 0.0)),
            EventOutcome(parent=0, parent_cont=1, cont_rewards=(0.0, 0.0)),
            EventOutcome(parent=1, parent_cont=0, cont_rewards=(1.0, 1.0)),
        ]

        assert event_weights(events, credit="subtree") == [0.25, 0.5, 1.0]

    def test_a_review_again_at_the_same_checkpoint_does_not_credit_the_replaced_turn(self) -> None:
        """Without a host continuation (checkpoint re-review), the new turn replaces the old one in its context."""
        events = [
            EventOutcome(parent=None, parent_cont=None, cont_rewards=(0.0, 0.0)),
            EventOutcome(parent=0, parent_cont=None, cont_rewards=(1.0, 1.0)),
        ]

        assert event_weights(events, credit="subtree") == [0.0, 1.0]

    def test_a_continuation_reward_is_never_lowered_by_its_subtree(self) -> None:
        events = [
            EventOutcome(parent=None, parent_cont=None, cont_rewards=(1.0, 0.0)),
            EventOutcome(parent=0, parent_cont=0, cont_rewards=(0.0, 0.0)),
        ]

        assert event_weights(events, credit="subtree") == [0.5, 0.0]

    def test_unknown_credit_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="credit"):
            event_weights([], credit="followed")


class TestFixedCoefficients:
    def test_coefficients_carry_the_fixed_token_normalizers(self) -> None:
        coef = fixed_coefficients(m=4, k=2, s=4, c=64.0, c_h=32.0)

        assert coef.root == pytest.approx(1 / (4 * 64))
        assert coef.repair == pytest.approx(1 / (4 * 2 * 64))
        assert coef.reflect(0.5) == pytest.approx(0.5 / (4 * 32))
