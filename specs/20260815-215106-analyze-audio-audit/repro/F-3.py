"""Reproduce F-3: stage_context.py justifies bumping STAGE_VERSIONS["ast"]/["yamnet"] by
saying wrapper-shaped output changes happen because "the classifiers (attach phoneme
labels)". Grep across the package shows phoneme-producing functions
(phoneme_similarity, g2p_phonemes) live entirely in asr.py/harvesters.py, and stages.py
shows ast/yamnet only ever produce AudioSet scene-classification fragments
(ast_result, yamnet_result, scene_agreement) -- no phoneme output anywhere near them.

No model load, no network: pure source-text / AST inspection.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-3.py
"""

import inspect
import sys
from pathlib import Path

import senselab.audio.workflows.audio_analysis.stage_context as stage_context
import senselab.audio.workflows.audio_analysis.stages as stages

module_doc = inspect.getsource(stage_context)
justification = "the classifiers (attach\nphoneme labels)".replace("\n", " ")
normalized_source = " ".join(module_doc.split())
assert "attach" in normalized_source and "phoneme labels" in normalized_source, (
    "stage_context.py wording changed; re-check the STAGE_VERSIONS justification text"
)

# Find every function name in stages.py that produces output attributed to "ast" or "yamnet",
# and check whether "phoneme" appears anywhere in that module at all.
stages_src = inspect.getsource(stages)
phoneme_hits_in_stages = stages_src.count("phoneme")

pkg_dir = Path(stage_context.__file__).resolve().parent
phoneme_producing_files = set()
for py_file in pkg_dir.glob("*.py"):
    text = py_file.read_text()
    if "phoneme_similarity" in text or "g2p_phonemes" in text or "def phoneme" in text:
        phoneme_producing_files.add(py_file.name)

print(f"stage_context.py justification for bumping STAGE_VERSIONS['ast']/['yamnet']: "
      f"'...the classifiers (attach phoneme labels)'")
print(f"phoneme occurrences inside stages.py (where ast/yamnet fragments are built): {phoneme_hits_in_stages}")
print(f"files that actually define/produce phoneme output: {sorted(phoneme_producing_files)}")

if phoneme_hits_in_stages == 0 and "stage_context.py" not in phoneme_producing_files and phoneme_producing_files:
    print("DEFECT REPRODUCED: stage_context.py's justification names phoneme labels for "
          "ast/yamnet, but stages.py (where ast/yamnet write their fragments) never "
          "mentions phonemes, and phoneme production lives entirely in "
          f"{sorted(phoneme_producing_files)}.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: phoneme production found near ast/yamnet in stages.py.")
    sys.exit(1)
