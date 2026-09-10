"""Slice-0 self-check: the FoB join must forward-fill, drop pre-first, never
cross hives. Run: python tests/test_io.py  (no framework, asserts only)."""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from urban.io import join_fob, load_inspections, parse_audio_name  # noqa: E402


def dt(day, hour=0):
    return datetime(2021, 6, day, hour)


def test_parse_audio_name():
    # real UrBAN format: DD-MM-YYYY_HHhMM_HIVE-<tag>.wav ; tag = inspection Tag number
    assert parse_audio_name("12-06-2021_08h30_HIVE-3629.wav") == ("3629", datetime(2021, 6, 12, 8, 30))
    for bad in ("nope.wav", "2021-06-12_HIVE-6.wav", "12-06-2021_0830_HIVE-6.mp3", "12-06-2021_08h30_R2.wav"):
        try:
            parse_audio_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected fail-loud on {bad!r}")


def test_load_inspections_real_schema():
    # fob = Fob 1st + 2nd + 3rd (NaN->0); row without Fob 1st is dropped; joins by Tag number
    csv = (
        "Date,Tag number,Colony Size,Fob 1st,Fob 2nd,Fob 3rd,FoBrood,Queen status\n"
        "2021-06-22,3629,1,6.0,,,,QR\n"          # fob=6
        "2021-07-01,3631,2,10.0,10.0,,,QR\n"     # fob=20
        "2021-07-05,6,3,8.0,7.0,5.0,,QR\n"       # fob=20
        "2021-07-09,99,1,,,,,QR\n"               # no Fob 1st -> dropped
    )
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        fh.write(csv)
        path = fh.name
    got = load_inspections(path)
    assert got == [
        ("3629", datetime(2021, 6, 22), 6.0),
        ("3631", datetime(2021, 7, 1), 20.0),
        ("6", datetime(2021, 7, 5), 20.0),
    ], got


def test_forward_fill_last_at_or_before():
    insp = [("R2", dt(5), 5.0), ("R2", dt(19), 8.0)]
    clips = [
        ("R2", dt(4), "a"),    # before first -> drop
        ("R2", dt(5), "b"),    # exactly on inspection -> gets it (side=right-1)
        ("R2", dt(10), "c"),   # between -> 5.0
        ("R2", dt(25), "d"),   # after last -> 8.0
    ]
    got = {p: f for _, _, p, f in join_fob(clips, insp)}
    assert "a" not in got, "clip before first inspection must be dropped"
    assert got == {"b": 5.0, "c": 5.0, "d": 8.0}, got


def test_no_cross_hive_leak():
    insp = [("A", dt(1), 3.0)]              # only hive A ever inspected
    clips = [("A", dt(10), "a"), ("B", dt(10), "b")]
    out = join_fob(clips, insp)
    assert [(h, p) for h, _, p, _ in out] == [("A", "a")], out  # B never labeled


def test_future_label_never_used():
    insp = [("A", dt(20), 9.0)]            # only a future inspection exists
    clips = [("A", dt(10), "a")]
    assert join_fob(clips, insp) == [], "must not borrow a future inspection"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
