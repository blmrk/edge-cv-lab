# Footage sources

Raw footage is never committed (see `.gitignore`). This file is the record of what was used, so every
number and image in the repo can be traced to a source and a licence. One row per clip.

| File | Source and URL | Author | Licence | Downloaded | Used for | Shown publicly? |
|---|---|---|---|---|---|---|
| `sample.mp4` | MTID (Multi-View Traffic Intersection Dataset), infrastructure camera, frames `seq3-infra_0000001` to `0003199`: https://vap.aau.dk/mtid/ (downloaded from Kaggle `andreasmoegelmose/multiview-traffic-intersection-dataset`, the authors' official copy) | M. B. Jensen, A. Møgelmose, T. B. Moeslund (Aalborg University) | CC BY 4.0 | 2026-09-26 | lab demo (video profile), zone metrics | yes, with credit and citation |
| `vecteezy-6434705.mp4` | Vecteezy 6434705, "Busy traffic on the highway": https://www.vecteezy.com/video/6434705-busy-traffic-on-the-highway (downloaded from Kaggle `aniketdash7/sample-video`, file `vecteezy_busy-traffic-on-the-highway_6434705.mp4`, 78563821 bytes; the Apache 2.0 label on Kaggle is the uploader's, not the author's) | Bondeto ae (Vecteezy contributor) | Vecteezy Free License, attribution required | 2026-09-28 | phantom boxes case study: boxes between side-by-side vehicles | no, metrics and credit only |
| `pexels-5124507.mp4` (was `sample.mp4`; deleted) | Pexels 5124507, https://www.pexels.com/video/5124507/ (fetched from Kaggle `arunavfc11/indian-traffic-videos`, file `5124507-hd_1920_1080_30fps.mp4`, same byte size as on Pexels' file server) | TBD: confirm on the Pexels page (it blocks automated access) | Pexels License | 2026-09-25 | replaced: video-profile plumbing test only; handheld, so no zone metrics | no |
| `UA-DETRAC/MVI_40714.mp4`, `UA-DETRAC/MVI_40714.xml` | UA-DETRAC test sequence MVI_40714: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Fetched from Kaggle mirrors: `longhn0108/ua-detrac-video-test`, file `MVI_40714.mp4` (46317496 bytes, an uploader's mpeg4 encode of the sequence's frames); `sudharsannv/detrac-xml`, file `DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/MVI_40714.xml` (11609640 bytes); and, for the alignment check only, `img00001.jpg`, `img00590.jpg`, `img01180.jpg` from `sudharsannv/detrac` (`DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40714/`) | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirrors' labels are the uploaders' | 2026-10-02 | task C: boxes straddling side-by-side vehicles; straddle and visit ground truth from the annotations | no, metrics and citation only |
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

- `vecteezy-6434705.mp4`: stock footage, not a mounted traffic camera: fixed, elevated telephoto view (overpass
  height) looking obliquely down a divided multi-lane expressway, daytime, cars two and three abreast with motorcycles
  between them. 1920x1080 at 50 fps, 1500 frames, 30.0 s (`ffprobe`). No annotations.

- `UA-DETRAC/MVI_40714.mp4`: research data, metrics only. A fixed, elevated view down a divided road, daytime: the
  near-left carriageway is queued several vehicles abreast, buses among them; the right carriageway is sparse.
  960x540 at 25 fps, 1180 frames, 47.2 s (`ffprobe -count_frames`); decoded frame n matches original image n+1
  (docs/datasets.md). Annotations: per-frame boxes with track IDs and three ignored regions (`detrac_to_gt.py`).

## Citations

- UA-DETRAC: Wen, L., Du, D., Cai, Z., et al. (2020). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection
  and Tracking. Computer Vision and Image Understanding. https://doi.org/10.1016/j.cviu.2020.102907
- MOT17: Milan, Leal-Taixé, Reid, Roth, Schindler. "MOT16: A Benchmark for Multi-Object Tracking." arXiv:1603.00831.
- MTID: Jensen, M. B., Møgelmose, A., & Moeslund, T. B. (2020). Presenting the Multi-View Traffic Intersection Dataset
  (MTID): A Detailed Traffic-Surveillance Dataset. IEEE 23rd International Conference on Intelligent Transportation
  Systems. https://doi.org/10.1109/ITSC45102.2020.9294694
