"""What the r37 B/C/G sections expect of each of the 24 + 6 noted stems: labelled.txt -> labels.json."""

import json
import re
import sys

EXPECT = {
    ("00053adb", "word-color-stroop"): "no_other_speaker",
    ("004d42e9", "free-speech"): "no_other_speaker",
    ("016023f6", "cape-v-sentences"): "no_other_speaker",
    ("017b7678", "word-color-stroop"): "no_other_speaker",
    ("02624589", "cape-v-sentences"): "no_other_speaker",
    ("028943cf", "productive-vocabulary"): "no_other_speaker",
    ("02a473a2", "free-speech"): "no_other_speaker",
    ("06b0a943", "cape-v-sentences"): "no_other_speaker",
    ("06d289b0", "free-speech-v2"): "no_other_speaker",
    ("06d289b0", "harvard-sentences-list"): "no_other_speaker",
    ("07929b7b", "diadochokinesis-pataka"): "no_other_speaker",
    ("07929b7b", "productive-vocabulary"): "off_task_not_other",
    ("07929b7b", "cinderella-story"): "cleared",
    ("fac74f45", "random-item-generation"): "cleared",
    ("168c5e68", "random-item-generation"): "speech_before_task",
    ("2ba9a335", "random-item-generation"): "pass_not_withheld",
    ("9488c55d", "animal-fluency"): "pass_not_withheld",
    ("8ca6baff", "random-item-generation-v2"): "withheld_not_pass",
    ("061cec48", "maximum-phonation-time"): "masks_narrowed",
    ("020f813b", "word-color-stroop"): "other_speaker_kept",
    ("0482bc77", "respiration-and-cough-breath"): "other_speaker_kept",
}

stems = [line.strip() for line in open(sys.argv[1]) if line.strip()]
labels = {s: EXPECT.get((s[4:12], re.sub(r"(-\d+)+$", "", s.split("_task-")[1])), "review") for s in stems}
json.dump(labels, open(sys.argv[2], "w"), indent=0)
print(len(labels), sum(v != "review" for v in labels.values()))
