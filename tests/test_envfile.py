""".env reading: export isn't overridden, key values aren't returned."""

from __future__ import annotations

import os

import pytest

from callfactory.cli import main
from callfactory.envfile import find_env_file, load_env_file, parse_env_file


def write_env(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_parses_the_usual_shapes(tmp_path):
    f = write_env(tmp_path / ".env", "\n".join([
        "# a comment",
        "",
        "PLAIN=abc",
        "export EXPORTED=def",
        'DOUBLE="ghi"',
        "SINGLE='jkl'",
        "SPACED =  mno  ",
        "EMPTY=",
        "no_equals_sign",
        "=novalue",
    ]))
    parsed = parse_env_file(f)
    assert parsed == {
        "PLAIN": "abc", "EXPORTED": "def", "DOUBLE": "ghi",
        "SINGLE": "jkl", "SPACED": "mno", "EMPTY": "",
    }


def test_an_exported_variable_wins(tmp_path, monkeypatch):
    f = write_env(tmp_path / ".env", "CF_TEST_KEY=from-file\n")
    monkeypatch.setenv("CF_TEST_KEY", "from-export")
    path, loaded = load_env_file(f)
    assert path == f
    assert loaded == []                      # no variables newly set
    assert os.environ["CF_TEST_KEY"] == "from-export"


def test_loads_what_is_not_set_and_returns_names_only(tmp_path, monkeypatch):
    f = write_env(tmp_path / ".env", "CF_TEST_A=secret-a\nCF_TEST_B=secret-b\nCF_TEST_C=\n")
    monkeypatch.delenv("CF_TEST_A", raising=False)
    monkeypatch.delenv("CF_TEST_B", raising=False)
    _, loaded = load_env_file(f)
    assert loaded == ["CF_TEST_A", "CF_TEST_B"]        # C has an empty value and doesn't count
    assert "secret-a" not in str(loaded), "the key itself must not appear in the return value"
    assert os.environ["CF_TEST_A"] == "secret-a"
    for name in ("CF_TEST_A", "CF_TEST_B"):
        monkeypatch.delenv(name, raising=False)


def test_an_explicit_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "nope.env")


def test_searching_upwards_finds_a_parent_env(tmp_path):
    write_env(tmp_path / ".env", "CF_TEST_PARENT=1\n")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_env_file(deep) == tmp_path / ".env"


def test_no_env_anywhere_is_not_an_error(tmp_path):
    path, loaded = load_env_file(start=tmp_path / "empty")
    assert loaded == []
    assert path is None or path.is_file()


def test_generate_prints_names_not_values(tmp_path, capsys):
    main(["init-bank", "--bank-dir", str(tmp_path)])
    capsys.readouterr()
    env = write_env(tmp_path / "keys.env", "CF_TEST_SHOWN=super-secret-value\n")
    code = main([
        "generate", "--bank-dir", str(tmp_path), "--output", str(tmp_path / "run"),
        "--run-id", "r", "--num-samples", "2", "--seed", "3",
        "--min-rounds", "5", "--max-rounds", "5", "--env-file", str(env),
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "CF_TEST_SHOWN" in out
    assert "super-secret-value" not in out
    os.environ.pop("CF_TEST_SHOWN", None)


def test_generate_reports_a_bad_env_path(tmp_path, capsys):
    main(["init-bank", "--bank-dir", str(tmp_path)])
    capsys.readouterr()
    code = main([
        "generate", "--bank-dir", str(tmp_path), "--output", str(tmp_path / "run"),
        "--num-samples", "1", "--env-file", str(tmp_path / "missing.env"),
    ])
    assert code == 2
    assert "env file not found" in capsys.readouterr().err
