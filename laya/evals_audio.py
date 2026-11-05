"""The sound-necessity gate: does the answer depend on the recording actually being there?

`laya.evals` scores answers against labels. This asks the question a benchmark of accuracy cannot:
run every row twice, once as given and once with the recording removed, and require accuracy to
fall. A "sound" checkpoint whose accuracy barely moves when the audio is removed is a text model
with an expensive tower and a block the encoder ignores -- the documented default failure of
multimodal fusion, and the one failure that looks like success on a leaderboard.

The measurement is `laya.evals_media.ablation`, which is also what the vision gate uses, so a
vision report and a sound report are produced by the same arithmetic and their gains are comparable
numbers rather than two implementations that happen to agree today.

Neither the ablation nor the option-order gate needs an audio checkpoint to exist to be useful: run
against a text runner, the ablation reports a gain of zero, which is the correct answer for a text
model. Both answer every row on its own by default, so a gate number stays comparable to published
one-state-at-a-time results; `batch_size` re-answers with several rows per pass (grouped by question
set), and both passes are always batched the same way, so the comparison holds.
"""
from typing import Any, Dict, Optional

from .audio import AUDIO_KEYS
from .evals import Dataset
from .evals_media import ablation

# Re-exported so a caller writing a sound gate does not have to import the vision module for a
# function that never looks at a state.
from .evals_vision import option_order_flip_rate, permuted_questions  # noqa: F401


def _has_audio(state: Any) -> bool:
    return isinstance(state, dict) and any(k in state for k in AUDIO_KEYS)


def strip_audio(state: Any) -> Any:
    """`state` with its sound keys removed.

    A dict that carried only a recording becomes an empty string rather than an empty dict: an empty
    dict is still a JSON document to serialize, so the model would be asked the question against
    `{}` -- text it was never meant to see -- and the ablation would be measuring the wrong thing.
    """
    if not _has_audio(state):
        return state
    rest = {k: v for k, v in state.items() if k not in AUDIO_KEYS}
    return rest if rest else ""


def audio_ablation(runner, dataset: Dataset, *, min_gain: Optional[float] = None,
                   min_text_change: float = 0.01,
                   batch_size: int = 1) -> Dict[str, Any]:
    """How much of the accuracy depends on the recording actually being there.

    Runs every row twice -- as given, and with the sound stripped -- and reports accuracy on both
    sides per tag slice, plus the drop. `min_gain` turns the report into a verdict: the run fails
    when the drop is smaller than that.

    Rows that carry no sound at all are counted under `rows_text_only` and excluded from the drop:
    there is nothing to remove, so including them would dilute the number toward zero and let a
    collapsed model pass.

    `min_text_change` is the companion check -- a text-only row must be answered the same with and
    without the ablation, because the ablation is a no-op for it. A row that moves anyway means the
    pass itself is non-deterministic, and every other number in the report is then noise.
    """
    return ablation(runner, dataset, gate="audio-necessity", noun="audio", keys=AUDIO_KEYS,
                    min_gain=min_gain, min_text_change=min_text_change,
                    batch_size=batch_size)
