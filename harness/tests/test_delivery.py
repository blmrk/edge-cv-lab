from collections import Counter

from replay.delivery import expected_events, finished_seeds, reconcile


def test_reconcile_a_duplicate_cannot_mask_a_loss():
    k = ("naive", "exit")
    # 3 expected, 3 rows stored, but only 2 distinct payloads: one event lost, another stored twice
    assert reconcile(Counter({k: 3}), Counter({k: 3}), Counter({k: 2})) == {
        "expected": 3, "stored": 3, "lost": 1, "extra": 0, "duplicates": 1}


def test_reconcile_counts_payloads_beyond_the_scene_as_extra():
    k = ("naive", "enter")
    assert reconcile(Counter({k: 2}), Counter({k: 3}), Counter({k: 3}))["extra"] == 1


def test_expected_events_match_the_live_sim_for_seed_11():
    # seed 11 as stored by the running lab: 206 naive and 17 debounced (= ground truth) visits
    assert expected_events(11) == Counter({("naive", "enter"): 206, ("naive", "exit"): 206,
                                           ("debounced", "enter"): 17, ("debounced", "exit"): 17})


def test_finished_seeds_come_from_play_order_not_only_ground_truth():
    # the sim plays seeds in order: events for a later seed mean every earlier seed has finished
    assert finished_seeds(truth={11, 12, 14}, seen={11, 12, 13, 14, 15}) == [11, 12, 13, 14]  # 13's truth lost, 15 running
    assert finished_seeds(truth={11, 12, 14, 15}, seen={11, 12, 14, 15, 16}) == [11, 12, 13, 14, 15]  # 13 lost outright
    assert finished_seeds(truth=set(), seen={11}) == []  # first scene still running
