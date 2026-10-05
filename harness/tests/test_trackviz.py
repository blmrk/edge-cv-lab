# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""The fixed-camera GIF writer on tiny frames, and per-run fixes over a tiny clip. Skipped where the [viz] extras are not
installed (CI)."""
import json
import sys

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

from replay import trackviz  # noqa: E402
from replay.trackviz import W, draw_tag, overlay_rgba, save_fixed_camera_gif  # noqa: E402
from replay.viz import draw_zone  # noqa: E402


def decoded(path):
    gif, out = Image.open(path), []
    for i in range(gif.n_frames):
        gif.seek(i)
        out.append((np.asarray(gif.convert("RGB")).astype(int), gif.info["duration"]))
    return out


def test_noise_below_hold_repeats_the_frame_and_motion_is_drawn(tmp_path):
    road = np.full((16, 16, 3), 100, np.uint8)
    noisy = road.copy()
    noisy[:8] += 5                                    # codec noise, under the hold
    car = noisy.copy()
    car[4:8, 4:8] = (230, 40, 40)                     # a vehicle arrives
    save_fixed_camera_gif([road, noisy, car, car], tmp_path / "a.gif", fps=10)
    (first, d1), (second, d2) = decoded(tmp_path / "a.gif")   # repeats merge into one frame each
    assert (d1, d2) == (200, 200)                     # 1000 / fps per source frame
    assert (first == 100).all()                       # the noise never reached the GIF
    assert (second[4:8, 4:8] == (230, 40, 40)).all()
    assert (second[8:] == 100).all() and (second[:4] == 100).all()


def test_reserved_colours_survive_a_busy_palette(tmp_path):
    # real footage fills the palette with thousands of colours; tags cover a few pixels, so median cut gives them no
    # exact slot and maps them to a dull neighbour unless their entries are reserved
    rng = np.random.default_rng(0)
    frame = rng.integers(30, 220, (192, 192, 3), dtype=np.uint8)
    tags = [(240, 70, 70), (70, 240, 90), (80, 110, 240)]
    for i, c in enumerate(tags):
        frame[4 + 8 * i:8 + 8 * i, 4:8] = c
    save_fixed_camera_gif([frame], tmp_path / "t.gif", fps=10, keep=tags)
    (img, _), = decoded(tmp_path / "t.gif")
    for i, c in enumerate(tags):
        assert (img[4 + 8 * i:8 + 8 * i, 4:8] == c).all(), c


def box(x, colour, shape=(16, 16)):
    """An opaque RGBA overlay: one 2 px wide vertical box edge at column x."""
    ov = np.zeros((*shape, 4), np.uint8)
    ov[2:14, x:x + 2] = (*colour, 255)
    return ov


def test_a_moving_overlay_leaves_no_trail(tmp_path):
    # a green box edge over a green vehicle: the box colour is within the hold of the footage under it, so a hold that
    # compares drawn frames keeps last frame's edge where the footage did not change, a trail of old box edges
    vehicle = (90, 170, 90)
    edge = (100, 185, 95)                             # under 20 levels from the vehicle in every channel
    bg0 = np.full((16, 16, 3), vehicle, np.uint8)
    bg1 = bg0.copy()
    bg1[:, :8] += 3                                   # sensor noise, under the hold
    save_fixed_camera_gif([bg0, bg1], tmp_path / "m.gif", fps=10, keep=[edge], overlays=[box(4, edge), box(7, edge)])
    first, second = [img for img, ms in decoded(tmp_path / "m.gif") for _ in range(ms // 100)]   # repeats unmerged
    assert (first[2:14, 4:6] == edge).all()
    assert (second[2:14, 4:6] == vehicle).all()       # where it was: the held footage, not the old edge
    assert (second[2:14, 7:9] == edge).all()          # the box where it is now, exact


def test_overlays_keep_the_hold_on_static_footage(tmp_path):
    road = np.full((16, 16, 3), 100, np.uint8)
    noisy = road.copy()
    noisy[8:] += 6                                    # codec noise, under the hold, partly under the box
    tag = (240, 70, 70)
    save_fixed_camera_gif([road, noisy, noisy], tmp_path / "s.gif", fps=10, keep=[tag], overlays=[box(4, tag)] * 3)
    (img, ms), = decoded(tmp_path / "s.gif")          # the three frames are identical, so one frame
    assert ms == 300
    assert (img[2:14, 4:6] == tag).all()
    img[2:14, 4:6] = 100
    assert (img == 100).all()                         # the noise never reached the GIF


def test_footage_never_takes_an_overlay_colour(tmp_path):
    # a green vehicle with no box on it: two footage entries leave its few pixels no entry of their own, and the
    # nearest entry is a reserved tag green, so the vehicle shows bright tag-green speckles unless footage maps to
    # footage entries
    tag = (107, 240, 80)
    frame = np.full((32, 32, 3), 100, np.uint8)
    frame[16:] = 30
    frame[2:4, 2:4] = (120, 225, 95)                  # four pixels of green livery
    save_fixed_camera_gif([frame], tmp_path / "p.gif", fps=10, colors=3, keep=[tag], overlays=[box(20, tag, (32, 32))])
    (img, _), = decoded(tmp_path / "p.gif")
    assert (img[2:14, 20:22] == tag).all()            # the box: exact
    assert not (img[2:4, 2:4] == tag).all(axis=2).any()   # the livery: never the tag's colour


def busy_road(seed, shape=(48, 48)):
    """Grey footage over many levels, a little sensor noise per channel."""
    rng = np.random.default_rng(seed)
    grey = rng.integers(40, 200, shape)[..., None] + rng.integers(-4, 5, (*shape, 3))
    return np.clip(grey, 0, 255).astype(np.uint8)


def test_a_rare_saturated_vehicle_keeps_its_colour(tmp_path):
    # an orange vehicle front in one frame of eight: median cut splits the many greys first and merges its few pixels
    # into a grey, and footage may not borrow the reserved tag orange, so the footage palette must give it an entry
    tag, orange = (240, 123, 80), (225, 130, 60)
    frames = [busy_road(0) for _ in range(8)]
    frames[0][4:10, 4:10] = orange
    ov = np.zeros((48, 48, 4), np.uint8)
    ov[30:44, 40:42] = (*tag, 255)                    # a tag-orange box elsewhere
    save_fixed_camera_gif(frames, tmp_path / "o.gif", fps=10, colors=32, keep=[tag], overlays=[ov] * 8)
    img = decoded(tmp_path / "o.gif")[0][0]
    assert np.abs(img[4:10, 4:10] - orange).max() <= 24   # orange, not grey-brown
    assert not (img[4:10, 4:10] == tag).all(axis=2).any()


def test_a_translucent_zone_fill_keeps_its_tint(tmp_path):
    # the zone fill blends 10% blue into the road under it: those pixels are footage (alpha under half), so they map
    # to footage entries only, and a palette built from the untinted footage would show them grey
    fill, tag = (60, 190, 230), (240, 123, 80)
    road = busy_road(1)
    ov = np.zeros((48, 48, 4), np.uint8)
    ov[:, 24:] = (*fill, 26)                          # the zone over the right half
    ov[2:14, 4:6] = (*tag, 255)
    save_fixed_camera_gif([road] * 8, tmp_path / "z.gif", fps=10, colors=16, keep=[tag], overlays=[ov] * 8)
    (img, _), = decoded(tmp_path / "z.gif")

    def tint(shown):                                  # mean blue minus red, against the footage alone
        r, _, b = (shown[:, 24:].astype(int) - road[:, 24:]).reshape(-1, 3).mean(axis=0)
        return b - r

    a = ov[..., 3:].astype(int)
    exact = (road.astype(int) * (255 - a) + ov[..., :3].astype(int) * a + 127) // 255
    assert tint(img) >= tint(exact) / 2               # the fill reads blue, not road grey


def test_overlay_drawn_on_black_and_white_composites_like_drawing_on_the_footage():
    # the zone fill is blended into what is under it; split into RGBA it must blend the same over held footage
    rng = np.random.default_rng(1)
    footage = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    poly = [(5, 5), (50, 8), (45, 35), (8, 30)]

    def draw(img):
        draw_zone(img, poly)
        cv2.rectangle(img, (20, 12), (34, 24), (70, 240, 90), 3)
        cv2.putText(img, "#7", (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)
        return img

    rgba = overlay_rgba(draw(np.zeros_like(footage)), draw(np.full_like(footage, 255)))
    a = rgba[..., 3:].astype(int)
    composite = (footage.astype(int) * (255 - a) + rgba[..., :3].astype(int) * a + 127) // 255
    assert np.abs(composite - draw(footage.copy())).max() <= 2
    assert (rgba[13:24, 21:34, 3] < 255).any()        # the fill shows the footage through it
    assert (rgba[11:14, 19:36, 3] == 255).all()       # the box edge is opaque


def tag_rows(top, by1, width=W):
    img = np.zeros((600, width, 3), np.uint8)
    draw_tag(img, 100, by1, 61, (70, 240, 90), top)
    rows = np.flatnonzero((img[:, 100] == (70, 240, 90)).all(axis=1))
    return rows.min(), rows.max()


def test_a_tag_sits_on_the_box_top_edge():
    assert tag_rows(0, 300)[1] == 300                 # the tag's bottom row is the box's top edge
    assert tag_rows(100, 300)[1] == 300               # crop from row 100: same


def test_a_tag_whose_box_top_is_under_the_header_goes_just_below_it():
    header = 40                                       # the strip header, at W px wide
    assert tag_rows(0, 10)[0] == header               # box top under the header
    assert tag_rows(100, 50)[0] == 100 + header       # box cut by the crop top
    assert tag_rows(0, header + 5)[0] == header       # top edge visible, but no room for the tag above it
    assert tag_rows(100, 50, width=1024)[0] == 100 + 32   # a 1024 px clip is scaled up to W: header 32 px here


FRAMES = 30


def scene(tmp_path, boxes):
    """A 160 x 120 clip of FRAMES grey frames at 10 fps, a zone over most of it, and the same detections, (bbox, score)
    of class car, on every frame, or, given a function of the frame, the detections it returns. Returns the trackviz
    arguments for them."""
    clip = str(tmp_path / "clip.avi")
    out = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(FRAMES):
        out.write(np.full((120, 160, 3), 90, np.uint8))
    out.release()
    at = boxes if callable(boxes) else lambda f: boxes
    (tmp_path / "dets.jsonl").write_text("".join(
        json.dumps({"frame": f, "ts_ms": 100 * f, "bbox": bbox, "score": score, "cls": "car"}) + "\n"
        for f in range(FRAMES) for bbox, score in at(f)))
    (tmp_path / "zone.json").write_text(json.dumps({"polygon": [[5, 5], [155, 5], [155, 115], [5, 115]]}))
    return ["--video", clip, "--dets", str(tmp_path / "dets.jsonl"), "--zone", str(tmp_path / "zone.json"),
            "--out", str(tmp_path / "out.gif"), "--width", "160"]


def render(monkeypatch, capsys, *args) -> list[str]:
    """trackviz's printed totals, one line per run."""
    monkeypatch.setattr(sys, "argv", ["trackviz", *args])
    trackviz.main()
    return capsys.readouterr().out.splitlines()[:-1]  # the last line names the file written


def fed(monkeypatch) -> list[list[int]]:
    """Per tracker, in --trackers order: how many detections it was given on each frame."""
    real, out = trackviz.create, []

    def create(name, **params):
        tracker, counts = real(name, **params), []
        out.append(counts)

        class Counting:
            def update(self, f, ts, ds):
                counts.append(len(ds))
                return tracker.update(f, ts, ds)
        return Counting()
    monkeypatch.setattr(trackviz, "create", create)
    return out


def test_the_zone_rule_is_per_run(tmp_path, monkeypatch, capsys):
    # a box that never moves, inside the zone: a lane marking scored as a car. The run without the rule counts it
    args = scene(tmp_path, [([60, 40, 100, 80], 0.9)])
    assert render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "greedy_iou", "--min-travel-px", "0", "30") \
        == ["greedy_iou: IDs 1, visits closed 1", "greedy_iou + zone rule 30 px: IDs 1, visits closed 0"]


@pytest.mark.parametrize("filtered_run", [0, 1])     # either order: a run's filtered frame must not leak to the next
def test_the_second_box_filter_is_per_run_and_before_the_tracker(tmp_path, monkeypatch, capsys, filtered_run):
    # a car box with a lower-scoring car box wholly inside it, on every frame: a second box on one vehicle
    args = scene(tmp_path, [([40, 30, 120, 90], 0.9), ([50, 40, 80, 70], 0.5)])
    given = fed(monkeypatch)
    filters = ["none", "none"]
    filters[filtered_run] = "contain090_same"
    lines = render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "greedy_iou",
                   "--labels", "a", "b", "--filters", *filters)
    seen = [[2] * FRAMES, [2] * FRAMES]
    seen[filtered_run] = [1] * FRAMES
    assert given == seen                                # only the filtered run's tracker misses the inner box
    said = ["a: IDs 2, visits closed 2", "b: IDs 2, visits closed 2"]
    said[filtered_run] = "ab"[filtered_run] + " + contain090_same: IDs 1, visits closed 1"
    assert lines == said


def test_without_the_new_options_runs_are_as_before(tmp_path, monkeypatch, capsys):
    args = scene(tmp_path, [([40, 30, 120, 90], 0.9), ([50, 40, 80, 70], 0.5)])
    given = fed(monkeypatch)
    assert render(monkeypatch, capsys, *args, "--trackers", "greedy_iou") == ["greedy_iou: IDs 2, visits closed 2"]
    assert given == [[2] * FRAMES]


def test_both_fixes_on_one_run_are_both_named(tmp_path, monkeypatch, capsys):
    args = scene(tmp_path, [([40, 30, 120, 90], 0.9), ([50, 40, 80, 70], 0.5)])
    assert render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "--labels", "IoU",
                  "--filters", "contain090_same", "--min-travel-px", "30") \
        == ["IoU + contain090_same + zone rule 30 px: IDs 1, visits closed 0"]


GREY = (200, 200, 200)                                # a reserved overlay colour over footage
TAGS = [trackviz._palette(i)[::-1] for i in range(trackviz.TAG_COLOURS)]   # the ID colours, RGB


def tag_row(img, y, x1, x2):
    """(pixels exactly grey, pixels exactly an ID colour) on row y from x1 to x2 of a decoded GIF frame."""
    row = img[y, x1:x2]
    return (row == GREY).all(axis=1).sum(), sum((row == c).all(axis=1).sum() for c in TAGS)


def tag_width(text):
    """Columns of a footage tag reading `text`, both rectangle edges included."""
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0] + 8 + 1


@pytest.mark.parametrize("ruled", [0, 1])              # either order: each strip asks its own run's counter
def test_a_track_the_zone_rule_holds_is_drawn_grey_and_tagged_held(tmp_path, monkeypatch, capsys, ruled):
    # a box that never moves, inside the zone: the run with the rule holds its enter back for the whole clip, and
    # shows it; the run without the rule draws it in its ID colour, as before
    args = scene(tmp_path, [([20, 50, 60, 90], 0.9)])
    travel = ["0", "0"]
    travel[ruled] = "30"
    lines = render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "greedy_iou", "--labels", "a", "b",
                   "--min-travel-px", *travel)
    said = ["a: IDs 1, visits closed 1", "b: IDs 1, visits closed 1"]
    said[ruled] = "ab"[ruled] + " + zone rule 30 px: IDs 1, visits closed 0"
    assert lines == said                                # labels and totals as without the grey
    last = decoded(tmp_path / "out.gif")[-1][0]
    for run in (0, 1):                                  # each strip is 120 rows; the tag's row under its text
        grey, coloured = tag_row(last, 120 * run + 49, 0, 160)
        if run == ruled:
            assert coloured == 0 and abs(grey - tag_width("#1 held")) <= 2, (run, grey, coloured)
        else:
            assert grey == 0 and abs(coloured - tag_width("#1")) <= 2, (run, grey, coloured)


def test_a_held_track_takes_its_colour_once_it_has_moved(tmp_path, monkeypatch, capsys):
    # one run with the rule: a box that never moves, and one that drives right 2 px a frame, 58 px by the last frame
    args = scene(tmp_path, lambda f: [([20, 50, 60, 90], 0.9), ([70 + 2 * f, 95, 100 + 2 * f, 115], 0.9)])
    assert render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "--min-travel-px", "30") \
        == ["greedy_iou + zone rule 30 px: IDs 2, visits closed 0"]
    frames = [img for img, _ in decoded(tmp_path / "out.gif")]
    first, last = frames[0], frames[-1]
    assert tag_row(first, 94, 70, 160)[1] == 0 and tag_row(first, 94, 70, 160)[0] > 80   # first frame: not moved yet
    grey, coloured = tag_row(last, 94, 128, 160)
    assert grey == 0 and coloured >= 28, (grey, coloured)           # moved 58 px: its ID colour
    grey, coloured = tag_row(last, 49, 0, 160)
    assert coloured == 0 and abs(grey - tag_width("#1 held")) <= 2  # never moved: still held


def test_a_vehicles_tag_is_drawn_over_a_held_tag_that_would_cover_it(tmp_path, monkeypatch, capsys):
    # #1 drives left 2 px a frame and ends with its tag inside the wider "#2 held" tag of a box that never moves;
    # the tracker reports #2 after #1, so drawing in its order would cover #1's tag: held tags go first
    from replay.schema import TrackBox

    class InOrder:                                      # ids in detection order, reported in that order every frame
        def update(self, f, ts, ds):
            return [TrackBox(f, ts, i + 1, tuple(d.bbox)) for i, d in enumerate(ds)]
    monkeypatch.setattr(trackviz, "create", lambda name, **params: InOrder())
    args = scene(tmp_path, lambda f: [([130 - 2 * f, 50, 160 - 2 * f, 90], 0.9), ([20, 50, 60, 90], 0.9)])
    render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "--min-travel-px", "30")
    last = decoded(tmp_path / "out.gif")[-1][0]
    grey, coloured = tag_row(last, 49, 72, 160)        # #1's tag starts at x=72, inside #2's held tag (20 to about 125)
    assert abs(coloured - tag_width("#1")) <= 2, (grey, coloured)


@pytest.mark.parametrize("extra, said", [
    (["--filters", "contain090_same"], ["--filters: 1 given for 2 trackers", "none for no filter"]),
    (["--min-travel-px", "30"], ["--min-travel-px: 1 given for 2 trackers", "0 for off"]),
    (["--filters", "none", "contain09"], ["--filters", "contain09", "contain090_same"]),   # names the known filters
])
def test_run_options_that_do_not_fit_stop_with_a_clear_message(tmp_path, monkeypatch, capsys, extra, said):
    args = scene(tmp_path, [([60, 40, 100, 80], 0.9)])
    with pytest.raises(SystemExit) as stop:
        render(monkeypatch, capsys, *args, "--trackers", "greedy_iou", "bytetrack", *extra)
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert all(s in err for s in said), err


def test_a_label_too_long_for_the_header_shrinks_to_end_before_the_stats():
    # the stats start at x=470; a label naming its fixes can run past that
    long = "bytetrack + contain090_same + zone rule 30 px"
    width = cv2.getTextSize(long, cv2.FONT_HERSHEY_SIMPLEX, trackviz.label_scale(long), 2)[0][0]
    assert 16 + width < 470
    assert trackviz.label_scale("greedy_iou:max_age=5, 0.17 s buffer") == 0.8   # the labels that fit are unchanged
