"""Adapter tests: verl parquet, TRL trainer_state, TRL completions, format sniffing."""

from __future__ import annotations

import json

import pytest

from rolloutscope.adapters import load, sniff


@pytest.fixture
def verl_dump(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rows = []
    for i in range(6):
        n = 4
        mask = [1] * n + [0] * 4
        scores = [0.0] * 8
        scores[n - 1] = float(i % 2)
        rows.append(
            {
                "uid": f"p{i // 2}",
                "index": i,
                "step": 1,
                "responses": list(range(8)),
                "response_mask": mask,
                "rollout_log_probs": [-0.7] * 8,
                "rm_scores": scores,
                "raw_prompt": [{"role": "user", "content": f"q{i}"}],
            }
        )
    path = tmp_path / "dump"
    path.mkdir()
    pd.DataFrame(rows).to_parquet(path / "rollout_dump.parquet", index=False)
    return str(path)


def test_sniff_detects_verl(verl_dump):
    assert sniff(verl_dump) == "verl"


def test_verl_adapter_reads_rewards_and_groups(verl_dump):
    batch = load(verl_dump)
    assert batch.framework == "verl"
    assert len(batch) == 6
    traces = batch.traces
    assert traces[0].group_id == "p0"
    assert traces[0].output_tokens == 4
    assert traces[0].reward == 0.0
    assert traces[1].reward == 1.0
    assert traces[0].mean_token_nll == pytest.approx(0.7, abs=1e-6)
    assert traces[0].prompt == "q0"


def test_trl_trainer_state_only(tmp_path):
    state = {
        "log_history": [
            {"step": 0, "entropy": 3.0, "completions/mean_length": 400, "frac_reward_zero_std": 0.1},
            {"step": 1, "entropy": 2.8, "completions/mean_length": 390, "frac_reward_zero_std": 0.2},
            {"step": 2, "entropy": 1.9, "completions/mean_length": 210, "frac_reward_zero_std": 0.4},
            {"step": 3, "entropy": 1.3, "completions/mean_length": 130, "frac_reward_zero_std": 0.5},
        ]
    }
    (tmp_path / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    assert sniff(str(tmp_path)) == "trl-state"
    batch = load(str(tmp_path))
    assert batch.framework == "trl"
    assert len(batch) == 0
    assert batch.metrics["entropy"][2] == 1.9
    assert batch.metrics["frac_reward_zero_std"][3] == 0.5


def test_trl_completions_group_by_prompt(tmp_path):
    rows = [
        {"step": 0, "prompt": "q1", "completion": "a", "reward": 1.0, "completion_tokens": 30},
        {"step": 0, "prompt": "q1", "completion": "b", "reward": 0.0, "completion_tokens": 32},
        {"step": 0, "prompt": "q2", "completion": "c", "reward": 1.0, "completion_tokens": 28},
        {"step": 0, "prompt": "q2", "completion": "d", "reward": 1.0, "completion_tokens": 29},
    ]
    path = tmp_path / "completions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert sniff(str(path)) == "trl"
    batch = load(str(path))
    assert len(batch) == 4
    assert batch.traces[0].group_id == batch.traces[1].group_id
    assert batch.traces[0].group_id != batch.traces[2].group_id
    assert batch.traces[0].output_tokens == 30
    assert batch.traces[2].reward == 1.0


def test_generic_jsonl_round_trip(tmp_path):
    from rolloutscope.schema import AgentTrace, Step, TokenUsage, TraceBatch

    batch = TraceBatch(
        traces=[
            AgentTrace(
                trace_id="a",
                group_id="g",
                step=0,
                steps=[Step(index=0, text="hi", token_usage=TokenUsage(output_tokens=3))],
                reward=1.0,
            )
        ],
        name="g",
    )
    path = tmp_path / "t.jsonl"
    batch.to_jsonl(str(path))
    assert sniff(str(path)) == "generic"
    back = load(str(path))
    assert back.traces[0].trace_id == "a"
    assert back.traces[0].steps[0].text == "hi"


def test_unknown_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(str(tmp_path / "missing"))
