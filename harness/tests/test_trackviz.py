# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Belmark Ray Nalugon (https://github.com/blmrk/edge-cv-lab)
"""The fixed-camera GIF writer on tiny frames. Skipped where the [viz] extras are not installed (CI)."""
import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

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
