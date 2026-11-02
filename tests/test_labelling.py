"""Human labelling interface: no machine-label leakage, round-by-round reveal, interruptible and resumable."""

from __future__ import annotations

import json

import pytest

from callfactory.labelling import (
    HumanLabel,
    _chunks,
    append_human_label,
    label_session,
    load_human_labels,
)


def make_convs(n=2, utterances=6):
    out = []
    for i in range(n):
        out.append({
            "conversation_id": f"run-{i:03d}",
            "utterances": [
                {"index": k, "speaker": "caller" if k % 2 == 0 else "receiver",
                 "text": f"line {k} of call {i}"}
                for k in range(utterances)
            ],
        })
    return out


class Script:
    """Feed inputs in order and collect the output."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.out: list[str] = []

    def read_line(self, prompt: str) -> str:
        self.out.append(prompt)
        if not self.answers:
            raise AssertionError(f"script ran out but the interface still wants input: {prompt!r}")
        return self.answers.pop(0)

    def write(self, text: str) -> None:
        self.out.append(text)

    @property
    def text(self) -> str:
        return "".join(self.out)


def test_reveals_one_round_at_a_time():
    assert list(_chunks(6, "round")) == [(0, 1), (2, 3), (4, 5)]
    assert list(_chunks(5, "round")) == [(0, 1), (2, 3), (4, 4)]
    assert list(_chunks(3, "utterance")) == [(0, 0), (1, 1), (2, 2)]
    with pytest.raises(ValueError):
        list(_chunks(4, "sideways"))


def test_the_flag_is_recorded_on_the_last_revealed_utterance(tmp_path):
    s = Script(["", "!", "y", "pressed for payment"])
    stats = label_session(tmp_path, make_convs(1), read_line=s.read_line, write=s.write)
    assert stats.labelled == 1
    label = load_human_labels(tmp_path)["run-000"]
    assert label.detection_index == 3        # pressed after the second round (index 2,3)
    assert label.detection_round == 2
    assert label.is_scam is True
    assert label.notes == "pressed for payment"


def test_no_alarm_is_distinguishable_from_not_asked(tmp_path):
    s = Script(["n", "n", ""])
    label_session(tmp_path, make_convs(1), read_line=s.read_line, write=s.write)
    label = load_human_labels(tmp_path)["run-000"]
    assert label.detection_index is None
    assert label.is_scam is False


def test_unsure_is_allowed(tmp_path):
    s = Script(["", "", "!", "u", ""])
    label_session(tmp_path, make_convs(1), read_line=s.read_line, write=s.write)
    assert load_human_labels(tmp_path)["run-000"].is_scam is None


def test_a_bad_verdict_is_re_asked(tmp_path):
    s = Script(["!", "maybe", "", "y", ""])
    label_session(tmp_path, make_convs(1), read_line=s.read_line, write=s.write)
    assert load_human_labels(tmp_path)["run-000"].is_scam is True
    assert "Please enter y / n / u" in s.text


def test_the_interface_never_shows_a_machine_label(tmp_path):
    convs = make_convs(1)
    s = Script(["", "", "", "y", ""])
    label_session(tmp_path, convs, read_line=s.read_line, write=s.write)
    shown = s.text.lower()
    for leaked in ("risk_score", "sufficient", "detect@", "earliest_detection",
                   "ground truth", "is_scam", "hard negative", "late onset", "scam_family"):
        assert leaked not in shown, f"labelling interface leaked {leaked!r}"


def test_whole_transcript_is_not_dumped_up_front(tmp_path):
    """Knowing the ending contaminates the "enough, having heard only the first k turns" judgement."""
    convs = make_convs(1, utterances=6)
    s = Script(["!", "y", ""])
    label_session(tmp_path, convs, read_line=s.read_line, write=s.write)
    # ! pressed in the first round; the next four utterances must not be printed
    assert "line 0 of call 0" in s.text and "line 1 of call 0" in s.text
    for k in (2, 3, 4, 5):
        assert f"line {k} of call 0" not in s.text


def test_skip_and_quit(tmp_path):
    s = Script(["s", "q"])
    stats = label_session(tmp_path, make_convs(3), read_line=s.read_line, write=s.write)
    assert stats.skipped == 1 and stats.labelled == 0 and stats.quit_early is True
    assert load_human_labels(tmp_path) == {}


def test_already_labelled_items_are_skipped_on_resume(tmp_path):
    # quit after labelling the first one, leaving the second for next time
    first = Script(["!", "y", "", "q"])
    label_session(tmp_path, make_convs(2), read_line=first.read_line, write=first.write)
    second = Script(["!", "n", ""])
    stats = label_session(tmp_path, make_convs(2), read_line=second.read_line, write=second.write)
    assert stats.already_done == 1 and stats.labelled == 1
    assert set(load_human_labels(tmp_path)) == {"run-000", "run-001"}
    assert "run-000" not in second.text


def test_relabelling_overwrites_by_appending(tmp_path):
    append_human_label(tmp_path, HumanLabel("c1", True, 3, 5))
    append_human_label(tmp_path, HumanLabel("c1", False, None, 9, notes="changed my mind"))
    label = load_human_labels(tmp_path)["c1"]
    assert label.is_scam is False and label.detection_index is None
    lines = (tmp_path / "human_labels.jsonl").read_text().splitlines()
    assert len(lines) == 2, "append-only: the old label stays, for auditing"


def test_only_filters_the_queue(tmp_path):
    s = Script(["!", "y", ""])
    stats = label_session(tmp_path, make_convs(3), read_line=s.read_line, write=s.write,
                          only=["run-001"])
    assert stats.labelled == 1
    assert set(load_human_labels(tmp_path)) == {"run-001"}


def test_an_empty_queue_says_so(tmp_path):
    s = Script([])
    stats = label_session(tmp_path, [], read_line=s.read_line, write=s.write)
    assert stats.labelled == 0
    assert "Nothing left to label" in s.text


def test_help_can_be_re_shown_without_advancing(tmp_path):
    s = Script(["?", "!", "y", ""])
    label_session(tmp_path, make_convs(1), read_line=s.read_line, write=s.write)
    assert s.text.count("this is it") >= 2      # once at the start + once on pressing ?
    assert load_human_labels(tmp_path)["run-000"].detection_index == 1


def test_labels_survive_a_corrupt_line(tmp_path):
    path = tmp_path / "human_labels.jsonl"
    path.write_text('{"conversation_id": "a", "is_scam": true, "detection_index": 1, '
                    '"revealed_through": 3}\nnot json at all\n', encoding="utf-8")
    labels = load_human_labels(tmp_path)
    assert set(labels) == {"a"}
