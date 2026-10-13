#!/bin/bash
# Writes the r36 single-job scripts from their r35 templates (r35 files are only read). Run once from $W.
# The sliced lanes (replay, bypass, review, opinion, cohort, refold, task_audio, rvec) are not sbatch files: they are
# slice_pool lanes of r36_slice.sh, chained by r36_link.sbatch.
set -euo pipefail
W=/orcd/scratch/bcs/002/satra/r18d_20261010
cd $W
conv() {  # $1 r35 template, $2 r36 output
  sed -e 's/\$R35/$R36/g' -e 's/\${R35/${R36/g' -e 's/common_r35\.sh/common_r36.sh/g' -e 's/job-name=r35-/job-name=r36-/' \
      -e 's#r18d_20261010/r35/logs/#r18d_20261010/r36/logs/#' -e 's/R35 = W + "\/r35"/R36 = W + "\/r36"/' "$1" > "$2.tmp.$$"
  mv "$2.tmp.$$" "$2"
}
for s in c_cohort_consistency f_merge f_check f_table f_copy f_finalize f_postcheck g_review_page g_verify; do
  conv ${s}_r35.sbatch ${s}_r36.sbatch
done
sed -i -e 's/^# r35 step 2: /# r36: /' c_cohort_consistency_r36.sbatch
# f_check: r36 paths and the replay rows too
sed -i -e 's/R35 = W + "\/r35"/R36 = W + "\/r36"/' -e 's/{R35}/{R36}/g' -e 's#r35/{step}_bad_stems#r36/{step}_bad_stems#' \
  -e 's#r35/<stage>#r36/<stage>#' -e 's/manifest_r35\.jsonl/manifest_r36.jsonl/' -e 's/per r35 stage/per r36 stage/' -e 's/in r35 logs/in r36 logs/' \
  -e 's/for step in ("rows_cohort", "rows_refold", "rows_task_audio")/for step in ("rows_replay", "rows_cohort", "rows_refold", "rows_task_audio")/' \
  f_check_r36.sbatch
grep -q '"rows_replay", "rows_cohort"' f_check_r36.sbatch
# g_review_page: r35 inherited ${R34:?}/review_stage from r34; r36 clears its own stage
sed -i -e 's#rm -rf "\${R34:?}/review_stage"#rm -rf "${R36:?}/review_stage"#' -e 's#r18d/r35 triage review#r18d/r36 triage review#' g_review_page_r36.sbatch
grep -q 'rm -rf "${R36:?}/review_stage"' g_review_page_r36.sbatch
# f_finalize: the description names r36 and what it did
/orcd/scratch/bcs/002/satra/senselab-r28/.venv/bin/python - <<'PY'
import re
p = "/orcd/scratch/bcs/002/satra/r18d_20261010/f_finalize_r36.sbatch"
s = open(p).read()
old_head = 'f"r18d/r35 at {commit[:8]}'
old_tail = "the whole corpus re-folded at r35, task audio,"
assert s.count(old_head) == 1 and s.count(old_tail) == 1, "f_finalize_r35 text moved; edit f_finalize_r36 by hand"
s = s.replace(old_head, 'f"r18d/r36 at {commit[:8]}')
s = s.replace(old_tail, "the whole corpus re-folded at r35, task audio; then (r36) loudness read as a voice task beyond its declared 'hey' (task_speech v{ts_version}: the word hey in any casing, punctuation, hyphenation or repetition is task text, every other lexical word is speech outside the task), the 1,602 loudness and loudness-v2 recordings replayed from TAXONOMY (packaged config, reviewer off; any reading left needs_reread re-read by the vLLM reviewer and the second opinion), COHORT re-applied over all 62,550 recordings under cohort_stage v{cs_version} (the loudness families read as non-lexical, no longer enrolled from, and compared with the speaker enrollment only on their task text), the whole corpus re-folded at r36, task audio,")
# the versions are read from the pinned checkout when finalize runs (common_r36.sh cds into it)
s = s.replace('d={"Name":', 'import re as _re\nver = lambda f: _re.search(r"^version:\\s*(\\S+)", open(f"src/senselab/audio/workflows/triage/data/{f}.yaml").read(), _re.M).group(1)\nts_version, cs_version = ver("task_speech"), ver("cohort_stage")\nd={"Name":', 1)
open(p, "w").write(s)
PY
grep -q 'r18d/r36 at' f_finalize_r36.sbatch
grep -q 'cohort_stage v{cs_version}' f_finalize_r36.sbatch
grep -q '^ts_version, cs_version = ' f_finalize_r36.sbatch
sed -i -e 's/^# PUBLISH: r18.s out\/ (now the r18d\/r34 stores)/# PUBLISH: r18'"'"'s out\/ (now the r18d\/r36 stores)/' f_copy_r36.sbatch
grep -n 'r35\|R35' c_cohort_consistency_r36.sbatch f_merge_r36.sbatch f_check_r36.sbatch f_table_r36.sbatch f_copy_r36.sbatch \
  f_finalize_r36.sbatch f_postcheck_r36.sbatch g_review_page_r36.sbatch g_verify_r36.sbatch | grep -v 'then (r35)\|re-folded at r35' || true
echo made
