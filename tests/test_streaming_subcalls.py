"""Integration tests for live streaming callbacks from REPL subcalls."""

from unittest.mock import patch

import rlm.core.rlm as rlm_module
from rlm import RLM
from tests.mock_lm import MockLM


def test_llm_query_streams_depth_one_tokens_from_repl_execution():
    """A real REPL llm_query call should emit live start/complete and depth>0 token events."""
    root_response = '```repl\nanswer = llm_query("child prompt")\nFINAL_VAR("answer")\n```'
    mock_lm = MockLM(responses=[root_response, "child answer"])
    tokens: list[tuple[int, str]] = []
    starts: list[tuple[int, str, str]] = []
    completes: list[tuple[int, str, str | None]] = []

    with patch.object(rlm_module, "get_client", return_value=mock_lm):
        rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": "mock-model"},
            max_depth=2,
            on_token=lambda depth, text: tokens.append((depth, text)),
            on_subcall_start=lambda depth, model, prompt_preview: starts.append(
                (depth, model, prompt_preview)
            ),
            on_subcall_complete=lambda depth, model, _duration, error: completes.append(
                (depth, model, error)
            ),
        )

        result = rlm.completion("context")

    assert result.response == "child answer"
    assert tokens == [
        (0, root_response),
        (1, "child answer"),
    ]
    assert starts == [(1, "mock-model", "child prompt")]
    assert completes == [(1, "mock-model", None)]
