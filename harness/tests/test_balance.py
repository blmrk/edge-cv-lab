from replay import balance
from replay.schema import TrackBox
from replay.zones import DebouncedZoneCounter, NaiveZoneCounter, ZoneEvent, run

SQUARE = [(100, 100), (500, 100), (500, 500), (100, 500)]


def _car(tid, frames, xs, y=300):
    """Footpoint at (x, y), 40 px wide; frame f at f * 100 ms."""
    return [TrackBox(f, f * 100, tid, (x - 20, y - 30, x + 20, y), 0.9, "car") for f, x in zip(frames, xs)]


def _scene():
    stays = _car(1, range(101), [min(10 * f, 300) for f in range(101)])  # drives in, still inside at the end
    lost = _car(2, range(31), [10 * f for f in range(31)], y=400)        # drives in, its track dies inside at 3 s
    return sorted(stays + lost, key=lambda b: (b.frame, b.track_id))


def test_occupancy_is_enters_minus_exits_up_to_each_frame():
    events = [ZoneEvent("enter", 1, 1000, 10), ZoneEvent("exit", 1, 3000, 30)]
    occ = balance.occupancy([(e.ts_ms, e) for e in events], [(f, f * 100) for f in range(40)])
    assert (occ[9], occ[10], occ[29], occ[30]) == (0, 1, 1, 0)


def test_the_debounced_counter_emits_an_enter_only_when_its_visit_closes():
    # stamped 1.5 s, when the car entered; emitted only at the end of the stream, where flush() releases it
    boxes = _scene()
    enters = [(at, e) for at, e in balance.emitted(DebouncedZoneCounter(SQUARE), boxes) if e.kind == "enter"]
    assert [(e.track_id, e.ts_ms, at) for at, e in enters] == [(2, 1500, 6100), (1, 1500, 10000)]


def test_true_occupancy_counts_annotated_vehicles_whose_anchor_is_inside():
    gt = _car(1, [0], [200]) + _car(2, [0], [300]) + _car(3, [0], [700]) + _car(1, [1], [200])
    assert balance.true_occupancy(gt, SQUARE) == {0: 2, 1: 1}


def test_open_visits_say_why_they_are_open():
    boxes = _scene()
    naive = balance.open_visits(run(NaiveZoneCounter(SQUARE, "centroid"), boxes), boxes, SQUARE, "centroid")
    assert sorted((r["track_id"], r["reason"]) for r in naive) == [(1, "inside at end"), (2, "lost inside")]
    debounced = balance.open_visits(run(DebouncedZoneCounter(SQUARE), boxes), boxes, SQUARE, "footpoint")
    assert [(r["track_id"], r["reason"]) for r in debounced] == [(1, "inside at end")]  # lost_ms closed track 2


def test_table_scores_each_counter_against_true_occupancy():
    boxes = _scene()
    truth = balance.true_occupancy(boxes, SQUARE)  # the tracks themselves as ground truth
    rows = {r["counter"]: r for r in balance.table(boxes, SQUARE, truth)}
    assert rows["naive, centroid"]["balance"] == 2 and rows["debounced"]["balance"] == 1
    assert rows["debounced"]["open_lost_inside"] == 0 and rows["naive, centroid"]["open_lost_inside"] == 1
    # the naive counter keeps the lost car inside for the rest of the clip; the debounced one closes it
    assert rows["naive, centroid"]["occupancy_mae"] > rows["debounced"]["occupancy_mae"]
    assert rows["debounced"]["end_counted"] == rows["debounced"]["end_true"] == 1
    # live, the debounced counter has emitted no enter for a car still inside, so it lags the truth more than after the fact
    assert rows["debounced"]["live_mae"] > rows["debounced"]["occupancy_mae"]
    # releasing the enter once dwell is proven lets the live view see the car that is still inside
    assert rows["debounced + zone rule 30 px + enter after dwell"]["live_mae"] < rows["debounced"]["live_mae"]


def test_occupancy_is_scored_only_over_the_annotated_frames():
    # frames past the last annotated one are not "empty zone", they are unannotated: leave them out, as TrackEval does
    boxes = _scene()
    truth = {f: n for f, n in balance.true_occupancy(boxes, SQUARE).items() if f <= 50}
    row = {r["counter"]: r for r in balance.table(boxes, SQUARE, truth)}["naive, centroid"]
    assert row["occupancy_mae"] == round(20 / 51, 3)  # frames 31-50 hold the lost car, 1 too many
    assert (row["end_counted"], row["end_true"]) == (2, 1)  # at frame 50, the last annotated one
