# Footage sources

Raw footage is never committed (see `.gitignore`). This file is the record of what was used, so every
number and image in the repo can be traced to a source and a licence. One row per clip.

| File | Source and URL | Author | Licence | Downloaded | Used for | Shown publicly? |
|---|---|---|---|---|---|---|
| `sample.mp4` | TBD | TBD | TBD | YYYY-MM-DD | lab demo, hero GIF | yes |
| `mot17-04.mp4` | MOTChallenge, https://motchallenge.net/data/MOT17 | Milan et al. | CC BY-NC-SA 3.0 | YYYY-MM-DD | tracker metrics only | no, metrics and citation only |

## Rules for this repo

1. **Anything visible** (GIFs, screenshots, the Grafana demo) comes only from footage with a licence that
   allows modification and republication: self-shot, CC0, CC BY, or a stock licence such as Pexels.
2. **Research datasets** (non-commercial or agreement-gated) are used for metrics only. Publish numbers
   and a citation, not frames.
3. Licence labels on third-party mirrors (Kaggle, Roboflow, Hugging Face re-uploads) are set by the
   uploader. Check the original authors' terms.
4. Not used, whatever the convenience: YouTube or city live-cam streams, anything from an employer or client.
5. Self-shot footage: fixed position, far enough that faces and number plates are not legible, or blur them.

## Per-clip notes

For each clip record: camera height and angle (estimate), duration, fps, resolution, lighting and weather,
who labelled the ground truth and how (`tools/label.html`, playback speed, single or double pass).

## Citations

- MOT17: Milan, Leal-Taixé, Reid, Roth, Schindler. "MOT16: A Benchmark for Multi-Object Tracking." arXiv:1603.00831.
