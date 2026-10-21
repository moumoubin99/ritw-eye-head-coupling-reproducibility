# Data access and local inputs

Obtain data directly from the official distributors and comply with their current access and license terms. No raw recordings or access tokens are included in this repository.

- Seattle access: https://www.projectaria.com/datasets/reading-in-the-wild/
- Seattle instructions and annotations: https://github.com/facebookresearch/reading_in_the_wild/tree/main/reading_in_the_wild_seattle
- Upstream analysis/data documentation: https://github.com/facebookresearch/reading_in_the_wild
- Columbus distributor: https://github.com/AIoT-MLSys-Lab/Reading-in-the-Wild-Columbus
- Columbus dataset card: https://huggingface.co/datasets/OSU-AIoT-MLSys-Lab/Reading-in-the-Wild-Columbus

The study's real-data analyses use the Seattle subset. The analysis cohort is regenerated locally from official annotations; no list of real source identifiers is distributed here. Columbus data access is documented for provenance and related validation work, not claimed as a required input to every included script.

Expected local inputs relative to the repository root:

```text
data/
  raw/seattle/<authorized-recording-id>/mps/
    eye_gaze/personalized_eye_gaze.csv   # general_eye_gaze.csv is the documented fallback
    slam/open_loop_trajectory.csv       # closed_loop_trajectory.csv is the fallback
    hand_tracking/hand_tracking_frames.jsonl
  ritw_repo/reading_in_the_wild_seattle/
    ritw_annotations.csv
    task16_annotations.csv
  ritw_meta/reading_in_the_wild_seattle/
    ritw_annotations.csv
```

`ritw_repo` is the loader's annotation location; the selected descriptive figure/table sources also use `ritw_meta`. Place the same authorized upstream annotation CSV in both locations, or link them locally. Do not publish those files or the links as part of this export.

The hand JSONL parser uses `tracking_timestamp_us` and each side's `existence_confidence`, wrist translation and quaternion. The distribution contains no source biometric time series. Official sequence links can expire; acquire any necessary download manifest yourself and keep it private. No downloader, token reader, signed URL or remote-host command is included in this public package.

After authorized local analysis, generated cohort and per-recording measurement files are required by the figure/table sources. The included `.gitignore` excludes common private inputs and recording-level outputs; inspect any new files before a public commit.
