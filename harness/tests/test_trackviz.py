"""The fixed-camera GIF writer on tiny frames. Skipped where the [viz] extras are not installed (CI)."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

from replay.trackviz import save_fixed_camera_gif  # noqa: E402


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
