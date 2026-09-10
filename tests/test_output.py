"""agent/cli/output.py: the file a turn's final answer lands in, since
copying a multi-line Rich panel out of a live terminal mangles box-drawing
borders and wrapped text (13.4's bug hunt) but a plain file survives intact.
"""
from agent.cli import output


def test_known_language_maps_to_its_conventional_extension():
    assert output._extension("python") == "py"
    assert output._extension("Python") == "py"  # case-insensitive
    assert output._extension("JavaScript") == "js"


def test_no_language_falls_back_to_markdown():
    assert output._extension(None) == "md"
    assert output._extension("") == "md"


def test_unrecognised_language_falls_back_to_itself_sanitised():
    assert output._extension("Fortran") == "fortran"
    assert output._extension("C#") == "c"  # non-alnum stripped, not silently .txt


def test_save_final_writes_the_exact_text_and_names_the_file_by_session_and_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "OUTPUT_DIR", tmp_path / "otto_output")

    path = output.save_final("abc123", 1, "def f():\n    return 1\n", "python")

    assert path == tmp_path / "otto_output" / "abc123-001.py"
    assert path.read_text() == "def f():\n    return 1\n"


def test_save_final_creates_the_output_directory_on_first_use(tmp_path, monkeypatch):
    target_dir = tmp_path / "does" / "not" / "exist" / "yet"
    monkeypatch.setattr(output, "OUTPUT_DIR", target_dir)

    output.save_final("s", 1, "hello", None)

    assert target_dir.is_dir()
    assert (target_dir / "s-001.md").read_text() == "hello"


def test_turn_number_is_zero_padded_and_distinguishes_repeated_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "OUTPUT_DIR", tmp_path)

    p1 = output.save_final("s", 1, "first", "python")
    p2 = output.save_final("s", 2, "second", "python")

    assert p1.name == "s-001.py"
    assert p2.name == "s-002.py"
    assert p1.read_text() == "first"
    assert p2.read_text() == "second"
