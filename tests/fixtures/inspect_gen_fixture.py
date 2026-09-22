"""Produce a REAL Inspect log containing a mix of correct/incorrect/unparseable.

The mockllm default output is unparseable for every sample, which exercises
only the invalid path. Driving mockllm with custom_outputs gives a genuine log
whose scores span C / I / and an unparseable response, so the adapter's
value mapping is tested against output Inspect actually wrote.
"""

import sys

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

sys.path.insert(0, ".")
from fixture_task import fixture_mcq  # noqa: E402

# fx_000 target=B, fx_001 target=A, fx_002 target=A
model = get_model(
    "mockllm/model",
    custom_outputs=[
        ModelOutput.from_content("mockllm/model", "ANSWER: B"),   # correct -> C
        ModelOutput.from_content("mockllm/model", "ANSWER: C"),   # wrong   -> I
        ModelOutput.from_content("mockllm/model", "I cannot."),   # unparseable
    ],
)

inspect_eval(
    fixture_mcq(),
    model=model,
    log_format="json",
    log_dir="scoredlogs",
    max_tokens=512,
    temperature=0.0,
    seed=42,
)
