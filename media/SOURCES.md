# Footage sources

Raw footage is never committed (see `.gitignore`). This file is the record of what was used, so every
number and image in the repo can be traced to a source and a licence. One row per clip.

| File | Source and URL | Author | Licence | Downloaded | Used for | Shown publicly? |
|---|---|---|---|---|---|---|
| `sample.mp4` | MTID (Multi-View Traffic Intersection Dataset), infrastructure camera, frames `seq3-infra_0000001` to `0003199`: https://vap.aau.dk/mtid/ (downloaded from Kaggle `andreasmoegelmose/multiview-traffic-intersection-dataset`, the authors' official copy) | M. B. Jensen, A. Møgelmose, T. B. Moeslund (Aalborg University) | CC BY 4.0 | 2026-09-26 | lab demo (video profile), zone metrics | yes, with credit and citation |
| `vecteezy-6434705.mp4` | Vecteezy 6434705, "Busy traffic on the highway": https://www.vecteezy.com/video/6434705-busy-traffic-on-the-highway (downloaded from Kaggle `aniketdash7/sample-video`, file `vecteezy_busy-traffic-on-the-highway_6434705.mp4`, 78563821 bytes; the Apache 2.0 label on Kaggle is the uploader's, not the author's) | Bondeto ae (Vecteezy contributor) | Vecteezy Free License, attribution required | 2026-09-28 | phantom boxes case study: boxes between side-by-side vehicles | no, metrics and credit only |
| `pexels-5124507.mp4` (was `sample.mp4`; deleted) | Pexels 5124507, https://www.pexels.com/video/5124507/ (fetched from Kaggle `arunavfc11/indian-traffic-videos`, file `5124507-hd_1920_1080_30fps.mp4`, same byte size as on Pexels' file server) | TBD: confirm on the Pexels page (it blocks automated access) | Pexels License | 2026-09-25 | replaced: video-profile plumbing test only; handheld, so no zone metrics | no |
| `UA-DETRAC/MVI_40714.mp4`, `UA-DETRAC/MVI_40714.xml` | UA-DETRAC test sequence MVI_40714: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Fetched from Kaggle mirrors: `longhn0108/ua-detrac-video-test`, file `MVI_40714.mp4` (46317496 bytes, an uploader's mpeg4 encode of the sequence's frames); `sudharsannv/detrac-xml`, file `DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/MVI_40714.xml` (11609640 bytes); and, for the alignment check only, `img00001.jpg`, `img00590.jpg`, `img01180.jpg` from `sudharsannv/detrac` (`DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40714/`) | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirrors' labels are the uploaders' | 2026-10-02 | task C: boxes straddling side-by-side vehicles; visit ground truth from the annotated tracks, and straddles by a fixed rule on the annotated boxes; the edge filter's live check, run locally through MediaMTX (metrics only) | no, metrics and citation only |
| `UA-DETRAC/MVI_40855.mp4`, `UA-DETRAC/MVI_40855.xml` | UA-DETRAC test sequence MVI_40855: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Fetched from the same Kaggle mirrors as MVI_40714: `longhn0108/ua-detrac-video-test`, file `MVI_40855.mp4` (25487055 bytes, an uploader's mpeg4 encode of the sequence's frames); `sudharsannv/detrac-xml`, file `DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/MVI_40855.xml` (11158567 bytes, delivered zipped); and, for the alignment check only, `img00001.jpg`, `img00545.jpg`, `img01090.jpg` from `sudharsannv/detrac` (`DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40855/`), kept in `UA-DETRAC/check-40855/` | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirrors' labels are the uploaders' | 2026-10-05 | task C, second clip: the second-box steps declared on a second annotated clip before any of them runs on it; visit ground truth from the annotated tracks | no, metrics and citation only |
| `UA-DETRAC/test-xml/` (32 XMLs), `UA-DETRAC/mirror-check/MVI_40714.xml`, `UA-DETRAC/mirror-check/MVI_40855.xml` | UA-DETRAC test annotations, no video or frames: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Fetched from the Kaggle mirror `sudharsannv/detrac-xml`, folder `DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/`: the XMLs of the 32 sequences in the third clip's candidate pool (docs/case-study-straddles.md, On a third clip), 161596999 bytes in all once unzipped (`wc -c`); and MVI_40714.xml and MVI_40855.xml again, into `UA-DETRAC/mirror-check/`, byte-identical (`cmp`) to the copies logged above, so the mirror is the one already logged. The two mirrors' listings are kept as `UA-DETRAC/listing-mp4.csv` and `UA-DETRAC/listing-xml.csv` | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirror's label is the uploader's (`kaggle` prints `License(s): unknown`) | 2026-10-06 | task C, third clip: the ranking that orders the candidates, from the annotations alone; metrics only | no, metrics and citation only |
| `UA-DETRAC/check-40863/` (3 JPEGs), `UA-DETRAC/MVI_40863.xml`; `UA-DETRAC/MVI_40863.mp4` not yet fetched | UA-DETRAC test sequence MVI_40863, the third clip's candidate 1: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Its first, middle and last original JPEGs, `img00001.jpg`, `img00835.jpg`, `img01670.jpg`, fetched from the Kaggle mirror the alignment checks' JPEGs come from, `sudharsannv/detrac` (`DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40863/`), for its zone and its alignment check, kept in `UA-DETRAC/check-40863/`; `MVI_40863.xml` is the file logged under `UA-DETRAC/test-xml/`, copied to `UA-DETRAC/`. The mp4, from `longhn0108/ua-detrac-video-test` (file `MVI_40863.mp4`), is not yet fetched | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirror's label is the uploader's (`kaggle` prints `License(s): unknown`) | 2026-10-06 | task C, third clip, candidate 1: its zone, drawn from the three JPEGs and the annotated tracks before any mp4 is fetched; visit ground truth from the annotated tracks | no, metrics and citation only |
| `UA-DETRAC/check-40742/` (3 JPEGs), `UA-DETRAC/MVI_40742.xml`; `UA-DETRAC/MVI_40742.mp4` not yet fetched | UA-DETRAC test sequence MVI_40742, the third clip's candidate 2: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Its first, middle and last original JPEGs, `img00001.jpg`, `img00827.jpg`, `img01655.jpg`, fetched from `sudharsannv/detrac` (`DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40742/`), for its zone and its alignment check, kept in `UA-DETRAC/check-40742/`; `MVI_40742.xml` is the file logged under `UA-DETRAC/test-xml/`, copied to `UA-DETRAC/`. The mp4, from `longhn0108/ua-detrac-video-test` (file `MVI_40742.mp4`), is not yet fetched | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirror's label is the uploader's (`kaggle` prints `License(s): unknown`) | 2026-10-06 | task C, third clip, candidate 2: its zone, drawn from the three JPEGs and the annotated tracks before any mp4 is fetched; visit ground truth from the annotated tracks | no, metrics and citation only |
| `UA-DETRAC/check-40864/` (3 JPEGs), `UA-DETRAC/MVI_40864.xml`; `UA-DETRAC/MVI_40864.mp4` not yet fetched | UA-DETRAC test sequence MVI_40864, the third clip's candidate 3: authors' page https://sites.google.com/view/daweidu/projects/ua-detrac (original site detrac-db.rit.albany.edu, now archived). Its first, middle and last original JPEGs, `img00001.jpg`, `img00757.jpg`, `img01515.jpg`, fetched from `sudharsannv/detrac` (`DETRAC-test-data/Insight-MVT_Annotation_Test/MVI_40864/`), for its zone and its alignment check, kept in `UA-DETRAC/check-40864/`; `MVI_40864.xml` is the file logged under `UA-DETRAC/test-xml/`, copied to `UA-DETRAC/`. The mp4, from `longhn0108/ua-detrac-video-test` (file `MVI_40864.mp4`), is not yet fetched | Wen et al. (UA-DETRAC) | CC BY-NC-SA 3.0, academic use only (original site, archived); the mirror's label is the uploader's (`kaggle` prints `License(s): unknown`) | 2026-10-06 | task C, third clip, candidate 3: its zone, drawn from the three JPEGs and the annotated tracks before any mp4 is fetched; visit ground truth from the annotated tracks | no, metrics and citation only |
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

- `UA-DETRAC/MVI_40855.mp4`: research data, metrics only. A fixed, elevated view over a large signalised intersection,
  daytime: the inbound approach is queued several vehicles abreast at a red light from the first frame and released
  partway through; on its left an outbound carriageway stays jammed for the whole clip, and a cross street runs through
  the foreground. 960x540 at 25 fps, 1090 frames, 43.6 s (`ffprobe -count_frames`); decoded frame n matches original
  image n+1 (docs/datasets.md). Annotations: per-frame boxes with track IDs, classes car, van and bus, and three ignored
  regions (`detrac_to_gt.py`; docs/case-study-straddles.md, Reproduce). Recorded at download (`wc -c`,
  `shasum -a 256`): `MVI_40855.mp4` 25487055 bytes, sha256
  `7f208102bf6a61d4593c791eeb61aacfa6aa8f4da2749b7bb9e4268e60399181`; `MVI_40855.xml` 11158567 bytes, sha256
  `577351bec22e368dda12d00a424d2f1ef6393f8dbc6bd3708b2601bbb8a5e599`; `check-40855/img00001.jpg`, `img00545.jpg` and
  `img01090.jpg` 65669, 63732 and 59484 bytes.

- `UA-DETRAC/test-xml/`: research data, metrics only. Annotations alone (per-frame boxes with track IDs and
  `vehicle_type`, ignored regions), no mp4 and no frames; read only by the third clip's ranking
  (docs/case-study-straddles.md, Reproduce), which also checks each file's size against the mirror's listing. Fetched
  by that Reproduce block, unzipped by it where the mirror delivers a file zipped. Recorded at download (`wc -c`,
  `shasum -a 256`):
  - `MVI_39031.xml` 2954404 bytes, sha256 `95104c003013c9d9ec5c3a9ee033275a4ea1c6054cfa84cbf7dfd761afe0f0e6`
  - `MVI_39051.xml` 725467 bytes, sha256 `c379907766c26646c08eb9716e28df055a64d7ab881aa985019ae9c0f5ccc863`
  - `MVI_39211.xml` 1227030 bytes, sha256 `86e17975608eaa2d260d078ad84a30f601759f6c5572ebca2c1fc4a417c46b55`
  - `MVI_39271.xml` 2829994 bytes, sha256 `c3259c9f1cd23369e82fe53aee3dec735d5e0d72ecefa1f815bd09fc0ae674a7`
  - `MVI_39311.xml` 7533158 bytes, sha256 `a5c4052c3ae287ca580a8539e489cb595f98deac855927cf4e37827c5fdf2b65`
  - `MVI_39361.xml` 3426515 bytes, sha256 `4e07b6e5eabd98531a1ca9bf24c2d58e9354d74de73bd835403eeb83fcf99dd1`
  - `MVI_39371.xml` 2020346 bytes, sha256 `bd435287f69bb1eca0b9365df088122c41e1bcdb9add7099a667fc227417f49e`
  - `MVI_39401.xml` 4201302 bytes, sha256 `ff98077a0e0c1788a6b298fb99a68ab81c9da57dc9001c086309e19f553cac17`
  - `MVI_39501.xml` 1674470 bytes, sha256 `d2e519ea3e6990b7eef5ff7fbba50803c5914a0b23d6d113baa028c1a2f0e6e0`
  - `MVI_39511.xml` 760473 bytes, sha256 `e82b1e8d02b1aa12a5fcf2937c68506d3148987200a6e2238baa3a71ebe40dd0`
  - `MVI_40701.xml` 5519833 bytes, sha256 `d2f317e05ff09ebc5a16232c4da78755f5d41c85955bcd57c416caa39e3a0d70`
  - `MVI_40742.xml` 9074694 bytes, sha256 `0364d4428215e98a59264a540d2b8f7218a3b0c488bc271d093fa53d439a0610`
  - `MVI_40743.xml` 4256770 bytes, sha256 `c8a8ee2411df9feaa1482bb3f9b39e9be52a0e7184a77e321a3b4a7c0a3e11c6`
  - `MVI_40761.xml` 7644314 bytes, sha256 `4271288834157dd274872015a03eae52d8d5311842e249a289ef0644af0e3928`
  - `MVI_40762.xml` 4913183 bytes, sha256 `0dfc15f8536bc4d3000ba36cda833854364851a390bd19c237fbbf5d01a8a224`
  - `MVI_40763.xml` 3134868 bytes, sha256 `2953c4fa771beb29bb163daf6e7d57bd9e364f5eaf6d1819e63fd9cba0398dea`
  - `MVI_40771.xml` 3011125 bytes, sha256 `8640fec3a2d73a0a1986b93064a5741f6a02d5495fb8c52c44f493ad5a1822ea`
  - `MVI_40772.xml` 4121383 bytes, sha256 `5eb7f2f1b261b5dad226b2aae59e984a45fad86ec4088791c51ee21156945b9f`
  - `MVI_40773.xml` 2134363 bytes, sha256 `10454921ec353499ef85ff0b83f8e67b9a692809d0d5be8eaeec4db45dc5f175`
  - `MVI_40774.xml` 1498086 bytes, sha256 `dc676601b1abd60dced781ee4ec85f730ee73c9d12c9f0b11d2b36580cbf133f`
  - `MVI_40775.xml` 3215392 bytes, sha256 `db6bb521552b2d665f5f0f910cdcdff46df78ecb643343418caa6c40af1719e5`
  - `MVI_40792.xml` 3966725 bytes, sha256 `141479733d4b252b0d8cf522a6e028300a7cbe8324e6905bab0e87bf23332d77`
  - `MVI_40793.xml` 6451842 bytes, sha256 `4722cf82de9c036745a7b4ab3e727a8a068a4f1a1e3d9238a44158c97b2a5242`
  - `MVI_40863.xml` 13091692 bytes, sha256 `6de301580ebf482c7af4c922c6445edcd9a04469064f0d70ef0f87f44df3596b`
  - `MVI_40864.xml` 16633019 bytes, sha256 `74de32395e1e23975932173e18b5b1cb2936bdc5d2da0e51a5df66e8972bffbb`
  - `MVI_40891.xml` 8991736 bytes, sha256 `a19547a2e65f977de244f8917db379bfb1f9b9218b94b1cd6359dbf00865a394`
  - `MVI_40892.xml` 12300866 bytes, sha256 `927b7b34f944e621e2b91f5ec7a777a5baece055ccc9007482de7dd99c5bb60e`
  - `MVI_40901.xml` 3320492 bytes, sha256 `3d102d914f35c0c6b7df978393456d22be1cce52d8507f313eb2add9e5861251`
  - `MVI_40902.xml` 3017991 bytes, sha256 `9bcf41176876996386769a2873e391449335a8f7250a5d950b093bd316b1b5b9`
  - `MVI_40903.xml` 7934478 bytes, sha256 `d3f724f9f1a79cad7d3f04f65546aa38dbd8c0d0839c215e018968eca4f28e7a`
  - `MVI_40904.xml` 4047759 bytes, sha256 `8f0491025240215f5045107b9cb664b0e9c9fdf09253182dffab4017412bc72b`
  - `MVI_40905.xml` 5963229 bytes, sha256 `7021fa5320b46034d5b928571b69ccdb3401c57b3337ae7f2eab162c2b26b27c`

- `UA-DETRAC/MVI_40863` (the third clip's candidate 1; mp4 not yet fetched): research data, metrics only. A fixed,
  elevated view along a multi-lane urban road, daytime, the road wet; all the annotated traffic runs left to right. A
  railing splits the near lanes from the far lanes; the near lanes stay queued for most of the clip, buses among them,
  and the road beyond the far lanes lies partly in the ignored regions and is not annotated outside them. Annotations: per-frame boxes with track IDs, classes
  bus, car, others and van, and five ignored regions; `detrac_to_gt.py` prints
  `runs/MVI_40863.gt.jsonl: 32634 boxes, 30 tracks, 69 boxes dropped in ignored regions` (docs/case-study-straddles.md,
  Reproduce). Recorded at download (`wc -c`, `shasum -a 256`): `check-40863/img00001.jpg` 77596 bytes, sha256
  `f1093ce63de9933c1cb34b2cc2a70571a8f27aac9da5c2b735e75632cbe27643`; `img00835.jpg` 77681 bytes, sha256
  `e7ea1c3355fdf117ed96a1b4bd749e394b13592ccbc72c953c58a38cd8cafdd9`; `img01670.jpg` 73921 bytes, sha256
  `3e3c8eaf24595bee562c0d13e445babe013b5903aeea98bf853e52b839732313`. `MVI_40863.xml`, copied from `test-xml/`, prints
  the size and sha256 logged there.

- `UA-DETRAC/MVI_40742` (the third clip's candidate 2; mp4 not yet fetched): research data, metrics only. A fixed,
  elevated view along a wide urban road at dusk, headlights on: a near carriageway and a middle carriageway split by a
  railing, both with traffic running from the far left down toward the camera's right, and beyond a second railing a
  road that lies in the ignored regions. The lanes beside the middle railing stay queued for most of the clip, buses
  among them, while the near carriageway's left lane keeps moving. Annotations: per-frame boxes with track IDs, classes
  bus and car, and five ignored regions; `detrac_to_gt.py` prints
  `runs/MVI_40742.gt.jsonl: 25053 boxes, 33 tracks, 87 boxes dropped in ignored regions`. Recorded at download
  (`wc -c`, `shasum -a 256`): `check-40742/img00001.jpg` 95592 bytes, sha256
  `a428083f0a685ca290e34b52be27dc4b3da2d54e43bb656958ef086e0c768962`; `img00827.jpg` 90443 bytes, sha256
  `05443fd27ed6e65aabc75f7f7d4c29861d4930cc5db5b7ffdda13db54cfa70a3`; `img01655.jpg` 85996 bytes, sha256
  `39a2eba56ea851363734c9a088888b62ee176cb8453d127ea767f9b7aa293d4c`. `MVI_40742.xml`, copied from `test-xml/`, prints
  the size and sha256 logged there.

- `UA-DETRAC/MVI_40864` (the third clip's candidate 3; mp4 not yet fetched): research data, metrics only. The view of
  MVI_40863: its first JPEG shows the scene of MVI_40863's last, and by its last JPEG the scene has shifted slightly up and to the left in the frame. Daytime, the road wet; all the annotated traffic runs left to right: the near lanes below the railing, where the lane beside the railing is queued until partway through the clip while the outer lane keeps moving; the far lanes, between the railing and a median; and a road beyond
  the median, queued for most of the clip. Unlike MVI_40863's, its ignored regions cover only the top of the frame and
  patches at its left, so the road beyond the median is annotated. Annotations: per-frame boxes with track IDs,
  classes bus, car, others and van, and three ignored regions; `detrac_to_gt.py` prints
  `runs/MVI_40864.gt.jsonl: 42966 boxes, 79 tracks, 1723 boxes dropped in ignored regions`. Recorded at download
  (`wc -c`, `shasum -a 256`): `check-40864/img00001.jpg` 73835 bytes, sha256
  `d5b558c4525121ae7b6dbe37754fab817772b4c05ef3d5e9bb21eaba9c6917d9`; `img00757.jpg` 77237 bytes, sha256
  `3ce835e763cf966e8ae98e7db245a6afac7acefd2b1ab2c1e36307a7fc35cc5f`; `img01515.jpg` 70359 bytes, sha256
  `02b5e6d962cb0a49bece3c071d61e97720a6bc510efff7b72542fcb042fa127c`. `MVI_40864.xml`, copied from `test-xml/`, prints
  the size and sha256 logged there.

## Citations

- UA-DETRAC: Wen, L., Du, D., Cai, Z., et al. (2020). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection
  and Tracking. Computer Vision and Image Understanding. https://doi.org/10.1016/j.cviu.2020.102907
- MOT17: Milan, Leal-Taixé, Reid, Roth, Schindler. "MOT16: A Benchmark for Multi-Object Tracking." arXiv:1603.00831.
- MTID: Jensen, M. B., Møgelmose, A., & Moeslund, T. B. (2020). Presenting the Multi-View Traffic Intersection Dataset
  (MTID): A Detailed Traffic-Surveillance Dataset. IEEE 23rd International Conference on Intelligent Transportation
  Systems. https://doi.org/10.1109/ITSC45102.2020.9294694
