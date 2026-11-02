"""Programmatic hard rules: leakage, alternation, length, repetition, anonymisation, label structure."""

from __future__ import annotations

import pytest

from callfactory.schema import PrefixLabel, Utterance, UtteranceLabel
from callfactory.validators import (
    ERROR,
    WARNING,
    annotate_repetition,
    find_repetitions,
    leaked_keys,
    placeholder_count,
    similarity,
    validate_conversation,
    validate_labels,
    validate_record_pair,
    validate_transcript_record,
)

from factories import make_conversation, make_scenario, make_task


def codes(report, severity=None):
    return [i.code for i in report.issues if severity is None or i.severity == severity]


# --------------------------------------------------------------------------- #
# Leakage (P2 / C1)
# --------------------------------------------------------------------------- #

def test_leaked_keys_searches_at_any_depth():
    assert leaked_keys({"a": {"b": [{"risk_score": 1}]}}) == {"risk_score"}
    assert leaked_keys({"utterances": [{"index": 0, "text": "hi"}]}) == set()
    assert leaked_keys([{"ground_truth": {}}, {"ok": 1}]) == {"ground_truth"}


def test_a_clean_transcript_passes():
    conv = make_conversation()
    report = validate_transcript_record(conv.transcript_record())
    assert report.ok and report.issues == []


def test_a_label_key_anywhere_in_the_transcript_is_an_error():
    record = make_conversation().transcript_record()
    record["utterances"][0]["risk_tier"] = "high"
    report = validate_transcript_record(record)
    assert not report.ok
    assert "leakage.label_in_transcript" in codes(report)
    # the same record is also caught by the extra-field check: two independent lines of defence
    assert "schema.extra_utterance_field" in codes(report)


def test_the_real_transcript_view_never_leaks():
    conv = make_conversation()
    assert leaked_keys(conv.transcript_record()) == set()
    # conversely, metadata of course contains labels
    assert leaked_keys(conv.metadata_record())


@pytest.mark.parametrize("mutate,code", [
    (lambda r: r.pop("conversation_id"), "schema.missing_field"),
    (lambda r: r.update(utterances=[]), "schema.no_utterances"),
    (lambda r: r["utterances"][0].update(speaker="narrator"), "schema.bad_speaker"),
    (lambda r: r["utterances"][0].update(text="   "), "schema.empty_text"),
    (lambda r: r["utterances"][0].update(index=9), "schema.bad_index"),
])
def test_schema_breakages_are_errors(mutate, code):
    record = make_conversation().transcript_record()
    mutate(record)
    report = validate_transcript_record(record)
    assert code in codes(report, ERROR)


# --------------------------------------------------------------------------- #
# Roles and length
# --------------------------------------------------------------------------- #

def test_role_alternation_must_start_with_the_caller():
    conv = make_conversation()
    conv.utterances[2].speaker = "receiver"
    report = validate_conversation(conv)
    assert "role.alternation" in codes(report, ERROR)


def test_too_short_is_an_error_and_too_long_is_a_warning():
    short = make_conversation(rounds=3)
    assert "length.too_short" in codes(validate_conversation(short), ERROR)
    # the fixture only has 5 lines, so any longer would hit the repetition check; hence lowering max_rounds rather than adding rounds
    long_ = make_conversation(rounds=5)
    report = validate_conversation(long_, max_rounds=4)
    assert "length.too_long" in codes(report, WARNING)
    assert report.ok, "too long is only a warning and must not block the data"


def test_a_one_word_turn_is_only_a_warning():
    conv = make_conversation()
    conv.utterances[1].text = "Right."
    report = validate_conversation(conv)
    assert "length.thin_utterance" in codes(report, WARNING)
    assert report.ok


# --------------------------------------------------------------------------- #
# Repetition
# --------------------------------------------------------------------------- #

def test_similarity_is_symmetric_and_bounded():
    a = "I do need to settle this today, the window closes at five"
    assert similarity(a, a) == 1.0
    assert similarity(a, "completely unrelated words here") < 0.1
    assert similarity("", "anything") == 0.0
    assert similarity(a, "I do need to settle this today, the window shuts at five") > 0.5


def test_repetitions_are_only_counted_within_one_speaker():
    line = "I do need to settle this today, the window closes at five."
    utterances = [
        Utterance(0, "caller", line),
        Utterance(1, "receiver", line),          # same as the caller's, but not a repetition
        Utterance(2, "caller", line),
    ]
    found = find_repetitions(utterances)
    assert set(found) == {2}
    assert found[2][0] == 0


def test_excessive_repetition_is_an_error():
    line = "I do need to settle this today, the window closes at five."
    conv = make_conversation()
    for utt in conv.utterances:
        utt.text = line
    report = validate_conversation(conv)
    assert "repetition.excessive" in codes(report, ERROR)


def test_annotate_repetition_writes_onto_the_utterance_labels():
    line = "I do need to settle this today, the window closes at five."
    conv = make_conversation()
    conv.utterances[0].text = line
    conv.utterances[2].text = line
    annotate_repetition(conv)
    labels = {u.index: u for u in conv.utterance_labels}
    assert labels[2].repetition_of == 0
    assert labels[2].repetition_similarity == 1.0
    assert labels[4].repetition_of is None


# --------------------------------------------------------------------------- #
# Anonymisation
# --------------------------------------------------------------------------- #

def test_a_long_digit_run_is_flagged():
    conv = make_conversation()
    conv.utterances[0].text = "Your reference is 4471 2233 9087, can you confirm?"
    assert "privacy.raw_digits" in codes(validate_conversation(conv), WARNING)


def test_an_off_vocabulary_placeholder_is_flagged():
    conv = make_conversation()
    conv.utterances[0].text = "I'm calling from [NATIONAL_BANK_OF_SOMEWHERE] about your file."
    assert "privacy.unknown_placeholder" in codes(validate_conversation(conv), WARNING)


def test_known_placeholders_are_fine():
    conv = make_conversation()
    conv.utterances[0].text = "This is [PERSON_1] at [COMPANY_A] about [ACCOUNT_REF]."
    assert "privacy.unknown_placeholder" not in codes(validate_conversation(conv))
    assert placeholder_count(conv.utterances[0].text) == 3


# --------------------------------------------------------------------------- #
# Label structure
# --------------------------------------------------------------------------- #

def test_a_label_borrowed_from_the_future_is_an_error():
    """The crux of the project: if the label at turn i contains information from after i, the latency metric is fake."""
    conv = make_conversation()
    conv.utterance_labels[2].label_from_index = 5
    assert "labels.future_leak" in codes(validate_labels(conv), ERROR)


def test_a_label_carried_forward_is_fine():
    conv = make_conversation()
    conv.utterance_labels[3].label_from_index = 2
    assert "labels.future_leak" not in codes(validate_labels(conv))


def test_a_disappearing_signal_is_an_error():
    conv = make_conversation()
    conv.prefix_labels[3].signals = []
    assert "labels.signal_regression" in codes(validate_labels(conv), ERROR)


def test_unordered_and_duplicated_prefix_labels_are_errors():
    conv = make_conversation()
    conv.prefix_labels[0].through_index = 99
    assert "labels.unordered" in codes(validate_labels(conv), ERROR)
    conv = make_conversation()
    conv.prefix_labels[1].through_index = conv.prefix_labels[0].through_index
    assert "labels.duplicate_prefix" in codes(validate_labels(conv), ERROR)


def test_bad_tier_and_out_of_range_score_are_errors():
    conv = make_conversation()
    conv.prefix_labels[0].risk_tier = "apocalyptic"
    conv.prefix_labels[1].risk_score = 4.0
    found = codes(validate_labels(conv), ERROR)
    assert "labels.bad_tier" in found
    assert "labels.score_out_of_range" in found


def test_detection_point_must_match_the_earliest_index():
    conv = make_conversation()
    conv.earliest_detection_index = 4
    for label in conv.utterance_labels:
        label.is_detection_point = label.index == 6
    assert "labels.detection_mismatch" in codes(validate_labels(conv), ERROR)


def test_detection_index_beyond_the_call_is_an_error():
    conv = make_conversation()
    conv.earliest_detection_index = 999
    assert "labels.detection_out_of_range" in codes(validate_labels(conv), ERROR)


def test_labels_stopping_early_are_only_a_warning():
    conv = make_conversation()
    conv.prefix_labels = conv.prefix_labels[:4]
    report = validate_labels(conv)
    assert "labels.incomplete" in codes(report, WARNING)
    assert report.ok


def test_an_unlabelled_conversation_still_validates():
    conv = make_conversation(with_labels=False)
    assert validate_conversation(conv).ok


# --------------------------------------------------------------------------- #
# Pairing the two files (C2)
# --------------------------------------------------------------------------- #

def test_a_matching_pair_passes():
    conv = make_conversation()
    report = validate_record_pair(conv.transcript_record(), conv.metadata_record())
    assert report.ok and not report.issues


def test_mismatched_ids_are_an_error():
    conv = make_conversation()
    meta = conv.metadata_record()
    meta["conversation_id"] = "someone-else-00001"
    report = validate_record_pair(conv.transcript_record(), meta)
    assert "pair.id_mismatch" in codes(report, ERROR)


def test_a_label_count_mismatch_is_a_warning():
    conv = make_conversation()
    meta = conv.metadata_record()
    meta["utterance_labels"] = meta["utterance_labels"][:3]
    report = validate_record_pair(conv.transcript_record(), meta)
    assert "pair.label_count" in codes(report, WARNING)
    assert report.ok
