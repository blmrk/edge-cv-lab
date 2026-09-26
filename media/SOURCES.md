# Footage sources

Raw footage is never committed (see `.gitignore`). This file is the record of what was used, so every
number and image in the repo can be traced to a source and a licence. One row per clip.

| File | Source and URL | Author | Licence | Downloaded | Used for | Shown publicly? |
|---|---|---|---|---|---|---|
| `sample.mp4` | MTID (Multi-View Traffic Intersection Dataset), infrastructure camera, frames `seq3-infra_0000001` to `0003199`: https://vap.aau.dk/mtid/ (downloaded from Kaggle `andreasmoegelmose/multiview-traffic-intersection-dataset`, the authors' official copy) | M. B. Jensen, A. Møgelmose, T. B. Moeslund (Aalborg University) | CC BY 4.0 | 2026-09-26 | lab demo (video profile), zone metrics | yes, with credit and citation |
| `pexels-5124507.mp4` (was `sample.mp4`; deleted) | Pexels 5124507, https://www.pexels.com/video/5124507/ (fetched from Kaggle `arunavfc11/indian-traffic-videos`, file `5124507-hd_1920_1080_30fps.mp4`, same byte size as on Pexels' file server) | TBD: confirm on the Pexels page (it blocks automated access) | Pexels License | 2026-09-25 | replaced: video-profile plumbing test only; handheld, so no zone metrics | no |
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

- `sample.mp4` (MTID): fixed pole-mounted camera over a signalised intersection, daytime, overcast, burned-in timestamp
  (2017-11-09). 1024x640 at 30 fps, 3199 frames (106.63 s): the range covered by the dataset's infrastructure
  annotations, which are kept locally under `media/candidates/mtid/annotations/`. Built from the frames with
  `ffmpeg -framerate 30 -start_number 1 -i seq3-infra_%07d.jpg -frames:v 3199 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart sample.mp4`.

## Citations

- MOT17: Milan, Leal-Taixé, Reid, Roth, Schindler. "MOT16: A Benchmark for Multi-Object Tracking." arXiv:1603.00831.
- MTID: Jensen, M. B., Møgelmose, A., & Moeslund, T. B. (2020). Presenting the Multi-View Traffic Intersection Dataset
  (MTID): A Detailed Traffic-Surveillance Dataset. IEEE 23rd International Conference on Intelligent Transportation
  Systems. https://doi.org/10.1109/ITSC45102.2020.9294694
