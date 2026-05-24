#!/usr/bin/env python3
"""Unit tests for ProposalVoteEnv."""

from __future__ import annotations

import numpy as np

from proposal_vote_env import Proposal, ProposalVoteConfig, ProposalVoteEnv, Vote


def test_payoff_table():
    c = ProposalVoteConfig(num_players=3)  # n = 4
    env = ProposalVoteEnv(c, seed=0)
    env.reset()

    cases = [
        (Proposal.A, [1, 1, 1], True, 4.0, -1.0),
        (Proposal.A, [0, 0, 0], False, -4.0, 1.0),
        (Proposal.A, [1, 0, 0], False, -4.0, 1.0),  # 1 accept, 2 reject → fail
        (Proposal.B, [1, 1, 1], True, 1.0, 1.0),
        (Proposal.B, [0, 0, 0], False, -1.0, -1.0),
    ]
    for prop, votes, accepted, exp_h, exp_p in cases:
        env.reset()
        env.step_propose(prop)
        _, rew, _, _, info = env.step_vote(np.array(votes, dtype=np.int32))
        assert info["accepted"] == accepted
        assert rew["host"] == exp_h
        assert np.all(rew["players"] == exp_p)


def test_players_never_see_proposal_in_obs():
    c = ProposalVoteConfig(num_players=5)
    env = ProposalVoteEnv(c, seed=1)
    env.reset()
    for prop in (Proposal.A, Proposal.B):
        env.reset()
        env.step_propose(prop)
        obs_vote = env._get_observations()
        env.step_vote(np.ones(c.num_players, dtype=np.int32))
        obs_after = env._get_observations()
        for obs in (obs_vote, obs_after):
            pobs = obs["players"]
            assert pobs["current"].shape == (c.num_players, c.current_player_dim)
            # history must not contain raw proposal (only host history has it)
            assert pobs["history"].shape[2] == c.hist_player_dim


def test_host_sees_proposal():
    env = ProposalVoteEnv(ProposalVoteConfig(num_players=4), seed=2)
    env.reset()
    env.step_propose(Proposal.B)
    host = env._get_observations()["host"]
    assert host["current"][1] == 0.0 and host["current"][2] == 1.0


def test_tie_passes_with_even_players():
    c = ProposalVoteConfig(num_players=4)  # n = 5
    env = ProposalVoteEnv(c, seed=9)
    env.reset()
    env.step_propose(Proposal.A)
    _, rew, _, _, info = env.step_vote(np.array([1, 1, 0, 0], dtype=np.int32))
    assert info["accepted"] is True
    assert rew["host"] == 5.0
    assert np.all(rew["players"] == -1.0)


def test_random_episode():
    c = ProposalVoteConfig(num_players=8, max_rounds=20)
    env = ProposalVoteEnv(c, seed=3)
    rng = np.random.default_rng(3)
    env.reset(seed=3)
    for _ in range(c.max_rounds):
        env.step_propose(int(rng.integers(0, 2)))
        votes = rng.integers(0, 2, size=c.num_players)
        _, _, term, _, info = env.step_vote(votes)
        assert info["proposal"] in (0, 1)
        if term:
            break
    assert env.round_idx == c.max_rounds


if __name__ == "__main__":
    test_payoff_table()
    test_players_never_see_proposal_in_obs()
    test_host_sees_proposal()
    test_random_episode()
    print("All tests passed.")
