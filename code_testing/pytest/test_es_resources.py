# test_es_resources.py
#
# Offline unit tests for worker-side machine sizing (src/es/resources.py) and
# the worker wiring that consumes it. Nothing here reads the host: every
# probe resources.py could make (psutil, os.cpu_count, os.sched_getaffinity,
# os.nice, the battery sensor) is injected, because the fleet is a 24-core
# 13900K, two 275HX laptops under WSL2 and a 10-core M4, and the box running
# the tests can only ever be one of them -- asserting against the real
# machine would make these tests pass for the wrong reason on three of four.
#
# No sockets, no emulator, no ROM, no sleeps: RollingRate and
# RollingRate both take an injectable clock for exactly this reason.

import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pytest

from es import resources
from es.resources import (
    NO_PSUTIL, RollingRate, Topology, apply_nice, detect_topology,
    on_battery, plan_procs,
)

MISSING = object()

# The fleet's real shapes. 13900K: 24 physical cores (8P+16E) behind 32
# logical. 275HX: 24 cores, no SMT. M4: 10 (4P+6E), Felipe's daily driver.
DESKTOP = Topology(logical_cpus=32, physical_cpus=24, platform="linux")
LAPTOP = Topology(logical_cpus=24, physical_cpus=24, platform="wsl")
M4 = Topology(logical_cpus=10, physical_cpus=10, platform="darwin")
# same desktop seen by a worker without psutil: hyperthreads look like cores
DESKTOP_BLIND = Topology(logical_cpus=32, physical_cpus=32, platform="linux",
                         physical_known=False)


# --------------------------------------------------------------------------
# Injectable fakes
# --------------------------------------------------------------------------

def fake_os(cpu_count=MISSING, affinity=MISSING, sys_platform="linux",
            uname_release=MISSING, environ=None, nice=MISSING):
    """A stand-in `os` module carrying only the attributes asked for, so a
    missing probe (no sched_getaffinity on macOS, no os.nice on Windows) is
    modelled by absence rather than by a flag."""
    ns = SimpleNamespace(sys_platform=sys_platform, environ=dict(environ or {}))
    if cpu_count is not MISSING:
        ns.cpu_count = lambda: cpu_count
    if affinity is not MISSING:
        ns.sched_getaffinity = lambda _pid: set(range(affinity))
    if uname_release is not MISSING:
        ns.uname = lambda: SimpleNamespace(release=uname_release)
    if nice is not MISSING:
        ns.nice = nice
    return ns


def fake_psutil(logical=MISSING, physical=MISSING, battery=MISSING, boom=False):
    """A stand-in `psutil`. `battery=MISSING` models an old psutil with no
    sensors_battery at all; `battery=None` models a desktop with no battery."""
    ns = SimpleNamespace()

    def cpu_count(logical=True):  # noqa: A002 -- psutil's own parameter name
        if boom:
            raise RuntimeError("psutil is having a day")
        value = ns._logical if logical else ns._physical
        if value is MISSING:
            raise AttributeError("not modelled")
        return value

    ns._logical, ns._physical = logical, physical
    ns.cpu_count = cpu_count
    if battery is not MISSING:
        ns.sensors_battery = lambda: battery
    return ns


class FakeClock:
    """Monotonic clock a test drives by hand. Callable, so it drops straight
    into the `clock=` seams."""

    def __init__(self, start=0.0):
        self.t = float(start)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def collector():
    """(warn callable, list it appends to)."""
    seen = []
    return seen.append, seen


# --------------------------------------------------------------------------
# plan_procs: the sizing policy, across the fleet's real shapes
# --------------------------------------------------------------------------

PLAN_CASES = [
    # auto (physical - reserve_cores) -- the default every machine gets
    ("desktop auto", DESKTOP, {}, 22),
    ("laptop auto", LAPTOP, {}, 22),
    ("m4 auto", M4, {}, 8),
    ("m4 auto, 4 reserved for a daily driver", M4, {"reserve_cores": 4}, 6),
    ("reserve 0 takes the whole box", DESKTOP, {"reserve_cores": 0}, 24),
    ("reserve == cores still leaves one worker", M4, {"reserve_cores": 10}, 1),
    ("reserve > cores still leaves one worker", M4, {"reserve_cores": 99}, 1),
    ("negative reserve reserves nothing", M4, {"reserve_cores": -3}, 10),
    ("garbage reserve reserves nothing", M4, {"reserve_cores": "two"}, 10),
    ("psutil-less desktop sizes off hyperthreads", DESKTOP_BLIND, {}, 30),

    # explicit --procs wins over the auto policy
    ("explicit beats reserve", M4, {"requested": 6}, 6),
    ("explicit beats share", M4, {"requested": 3, "cpu_share": 0.9}, 3),
    ("explicit may oversubscribe on purpose", M4, {"requested": 40}, 40),
    ("explicit 0 floors at one process", M4, {"requested": 0}, 1),
    ("explicit negative floors at one process", M4, {"requested": -5}, 1),
    ("explicit arrives as a CLI string", M4, {"requested": "12"}, 12),
    ("explicit float truncates", M4, {"requested": 6.9}, 6),
    ("unparseable explicit falls back to auto", M4, {"requested": "banana"}, 8),
    ("explicit bool is a bug, not a request", M4, {"requested": True}, 8),

    # --cpu-share: donate a fraction of the machine
    ("half a desktop", DESKTOP, {"cpu_share": 0.5}, 12),
    ("half an M4", M4, {"cpu_share": 0.5}, 5),
    ("all of an M4", M4, {"cpu_share": 1.0}, 10),
    ("share beats reserve", M4, {"cpu_share": 0.9, "reserve_cores": 8}, 9),
    ("share rounds half up, not to even", M4, {"cpu_share": 0.25}, 3),
    ("share rounds down below the half", DESKTOP, {"cpu_share": 0.1}, 2),
    ("a sliver still runs one process", M4, {"cpu_share": 0.01}, 1),
    ("zero share still runs one process", M4, {"cpu_share": 0.0}, 1),
    ("negative share still runs one process", M4, {"cpu_share": -1.0}, 1),
    ("share above 1 clamps to the machine", M4, {"cpu_share": 3.0}, 10),
    ("unparseable share falls back to auto", M4, {"cpu_share": "half"}, 8),
    ("nan share falls back to auto", M4, {"cpu_share": float("nan")}, 8),

    # --max-procs: the hard cap, applied last to every path
    ("cap trims auto", DESKTOP, {"max_procs": 4}, 4),
    ("cap trims explicit", M4, {"requested": 40, "max_procs": 8}, 8),
    ("cap trims share", DESKTOP, {"cpu_share": 1.0, "max_procs": 6}, 6),
    ("cap above the plan changes nothing", M4, {"max_procs": 99}, 8),
    ("cap of 0 is ignored, not obeyed", M4, {"max_procs": 0}, 8),
    ("garbage cap is ignored", M4, {"max_procs": "eight"}, 8),

    # degenerate topologies: a machine we could not read is still a worker
    ("zero-core topology floors at one", Topology(0, 0, "?"), {}, 1),
    ("None-core topology floors at one", Topology(None, None, "?"), {}, 1),
    ("single-core box", Topology(1, 1, "linux"), {}, 1),
    ("dual-core box, 2 reserved", Topology(2, 2, "linux"), {}, 1),
]


@pytest.mark.parametrize("label,topology,kwargs,expected",
                         PLAN_CASES, ids=[c[0] for c in PLAN_CASES])
def test_plan_procs_table(label, topology, kwargs, expected):
    assert plan_procs(topology, **kwargs) == expected


@pytest.mark.parametrize("topology,kwargs", [(t, k) for _l, t, k, _e in PLAN_CASES])
def test_plan_procs_never_returns_less_than_one(topology, kwargs):
    # a worker that plans 0 processes is a machine silently contributing
    # nothing while looking healthy in /status -- the one outcome to rule out
    assert plan_procs(topology, **kwargs) >= 1


def test_plan_procs_is_silent_without_a_logger(capsys):
    # pure by default: the coordinator/tests can call it in a loop
    plan_procs(M4, requested=0, max_procs=-1)
    assert capsys.readouterr().out == ""


def test_plan_procs_survives_a_non_callable_warn():
    assert plan_procs(M4, warn="not a function") == 8


def test_explicit_procs_fat_finger_hits_the_sanity_ceiling():
    """--procs 999 in an autostarted unit must clamp (4x physical), not OOM
    the machine. Legitimate oversubscription below the ceiling still obeys."""
    warns = []
    assert plan_procs(M4, requested=999, warn=warns.append) == 40
    assert any("clamping to 40" in w for w in warns)
    assert plan_procs(M4, requested=15, warn=warns.append) == 15


def test_plan_procs_warns_when_oversubscribing():
    warn, seen = collector()
    assert plan_procs(M4, requested=40, warn=warn) == 40
    assert any("exceeds" in m for m in seen)


def test_plan_procs_warns_when_flooring_to_one():
    warn, seen = collector()
    plan_procs(M4, requested=0, warn=warn)
    assert any("below 1" in m for m in seen)


def test_plan_procs_warns_when_the_cap_bites():
    warn, seen = collector()
    plan_procs(DESKTOP, max_procs=4, warn=warn)
    assert any("--max-procs" in m for m in seen)


def test_plan_procs_warns_when_physical_cores_are_guessed():
    warn, seen = collector()
    plan_procs(DESKTOP_BLIND, warn=warn)
    assert any("psutil" in m for m in seen)


def test_plan_procs_does_not_nag_about_psutil_when_procs_is_explicit():
    # the count came from a human; the core count was never used
    warn, seen = collector()
    plan_procs(DESKTOP_BLIND, requested=8, warn=warn)
    assert not any("psutil" in m for m in seen)


# --------------------------------------------------------------------------
# detect_topology: psutil is optional everywhere
# --------------------------------------------------------------------------

def test_detect_topology_prefers_psutil_physical_cores():
    topo = detect_topology(psutil_mod=fake_psutil(logical=32, physical=24),
                           os_mod=fake_os(cpu_count=32))
    assert (topo.logical_cpus, topo.physical_cpus, topo.physical_known) == (32, 24, True)


def test_detect_topology_without_psutil_assumes_logical_are_cores():
    topo = detect_topology(psutil_mod=NO_PSUTIL, os_mod=fake_os(cpu_count=32))
    assert (topo.logical_cpus, topo.physical_cpus, topo.physical_known) == (32, 32, False)


def test_detect_topology_survives_a_psutil_that_raises():
    topo = detect_topology(psutil_mod=fake_psutil(boom=True), os_mod=fake_os(cpu_count=10))
    assert (topo.logical_cpus, topo.physical_cpus, topo.physical_known) == (10, 10, False)


def test_detect_topology_survives_psutil_returning_none():
    # psutil.cpu_count(logical=False) is documented to return None on some platforms
    topo = detect_topology(psutil_mod=fake_psutil(logical=None, physical=None),
                           os_mod=fake_os(cpu_count=8))
    assert (topo.logical_cpus, topo.physical_cpus, topo.physical_known) == (8, 8, False)


@pytest.mark.parametrize("cpu_count", [None, 0, -4, "many", MISSING])
def test_detect_topology_floors_at_one_cpu(cpu_count):
    kwargs = {} if cpu_count is MISSING else {"cpu_count": cpu_count}
    topo = detect_topology(psutil_mod=NO_PSUTIL, os_mod=fake_os(**kwargs))
    assert (topo.logical_cpus, topo.physical_cpus) == (1, 1)


def test_detect_topology_respects_cpu_affinity():
    # a WSL2 slice (.wslconfig processors=20) or a taskset: cpu_count describes
    # a machine we are not allowed to fill
    topo = detect_topology(psutil_mod=fake_psutil(logical=32, physical=24),
                           os_mod=fake_os(cpu_count=32, affinity=20))
    assert (topo.logical_cpus, topo.physical_cpus) == (20, 20)


def test_detect_topology_affinity_never_inflates_the_count():
    topo = detect_topology(psutil_mod=NO_PSUTIL, os_mod=fake_os(cpu_count=8, affinity=64))
    assert topo.logical_cpus == 8


def test_detect_topology_survives_a_broken_affinity_probe():
    bad = fake_os(cpu_count=10)
    bad.sched_getaffinity = lambda _pid: (_ for _ in ()).throw(OSError("nope"))
    assert detect_topology(psutil_mod=NO_PSUTIL, os_mod=bad).logical_cpus == 10


PLATFORM_CASES = [
    ("darwin", MISSING, {}, "darwin"),
    ("linux", "6.8.0-generic", {}, "linux"),
    ("linux", "5.15.153.1-microsoft-standard-WSL2", {}, "wsl"),
    ("linux", MISSING, {"WSL_DISTRO_NAME": "Ubuntu-24.04"}, "wsl"),
    ("linux", MISSING, {"WSL_INTEROP": "/run/WSL/8_interop"}, "wsl"),
    ("win32", MISSING, {}, "win32"),
]


@pytest.mark.parametrize("platform,release,env,expected", PLATFORM_CASES)
def test_detect_topology_platform_label(platform, release, env, expected):
    topo = detect_topology(psutil_mod=NO_PSUTIL,
                           os_mod=fake_os(cpu_count=4, sys_platform=platform,
                                          uname_release=release, environ=env))
    assert topo.platform == expected


def test_detect_topology_reads_nothing_when_probes_are_hostile():
    hostile = SimpleNamespace(sys_platform="linux")
    hostile.cpu_count = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    topo = detect_topology(psutil_mod=NO_PSUTIL, os_mod=hostile)
    assert topo.logical_cpus == 1 and topo.platform == "linux"


# The psutil_mod=None path lazily imports psutil, which is the one probe that
# cannot be handed in as an argument. Patching sys.modules exercises both of
# its outcomes without the test depending on whether THIS box has psutil.

def test_detect_topology_lazily_imports_psutil_when_it_is_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil(logical=32, physical=24))
    topo = detect_topology(os_mod=fake_os(cpu_count=32))
    assert (topo.physical_cpus, topo.physical_known) == (24, True)


def test_detect_topology_degrades_when_psutil_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise
    topo = detect_topology(os_mod=fake_os(cpu_count=32))
    assert (topo.physical_cpus, topo.physical_known) == (32, False)


# --------------------------------------------------------------------------
# apply_nice: the real "stay usable" knob, and never fatal
# --------------------------------------------------------------------------

def test_apply_nice_returns_the_resulting_niceness():
    calls = []
    os_mod = fake_os(nice=lambda delta: calls.append(delta) or 10)
    assert apply_nice(10, os_mod=os_mod) == 10
    assert calls == [10]


def test_apply_nice_passes_zero_through_to_read_current_priority():
    assert apply_nice(0, os_mod=fake_os(nice=lambda delta: 5)) == 5


@pytest.mark.parametrize("boom", [PermissionError("needs root"), OSError("nope"),
                                  RuntimeError("weird libc")])
def test_apply_nice_degrades_to_none_when_the_os_refuses(boom):
    warn, seen = collector()

    def nice(_delta):
        raise boom

    assert apply_nice(-5, os_mod=fake_os(nice=nice), warn=warn) is None
    assert seen, "a refused renice must be reported, not swallowed"


def test_apply_nice_degrades_to_none_without_os_nice():
    # native Windows python has no os.nice; a worker there must still start
    warn, seen = collector()
    assert apply_nice(10, os_mod=fake_os(), warn=warn) is None
    assert any("os.nice" in m for m in seen)


@pytest.mark.parametrize("delta", [None, "ten", object(), float("nan")])
def test_apply_nice_rejects_non_integer_deltas(delta):
    called = []
    assert apply_nice(delta, os_mod=fake_os(nice=called.append)) is None
    assert called == [], "a garbage delta must never reach os.nice"


def test_apply_nice_handles_an_os_nice_that_returns_nothing():
    assert apply_nice(10, os_mod=fake_os(nice=lambda _d: None)) is None


# --------------------------------------------------------------------------
# on_battery: warn, never block
# --------------------------------------------------------------------------

@pytest.mark.parametrize("plugged,expected", [(False, True), (True, False), (None, None)])
def test_on_battery_reads_the_sensor(plugged, expected):
    sensor = SimpleNamespace(percent=57.0, power_plugged=plugged)
    assert on_battery(psutil_mod=fake_psutil(battery=sensor)) is expected


def test_on_battery_is_unknown_on_a_desktop_with_no_battery():
    assert on_battery(psutil_mod=fake_psutil(battery=None)) is None


def test_on_battery_is_unknown_without_psutil():
    assert on_battery(psutil_mod=NO_PSUTIL) is None


def test_on_battery_is_unknown_when_psutil_lacks_the_sensor():
    assert on_battery(psutil_mod=fake_psutil()) is None


def test_on_battery_is_unknown_when_the_sensor_raises():
    ps = SimpleNamespace()
    ps.sensors_battery = lambda: (_ for _ in ()).throw(RuntimeError("no acpi"))
    assert on_battery(psutil_mod=ps) is None


def test_on_battery_lazily_imports_psutil(monkeypatch):
    sensor = SimpleNamespace(power_plugged=False)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil(battery=sensor))
    assert on_battery() is True


def test_on_battery_is_unknown_when_psutil_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert on_battery() is None


# --------------------------------------------------------------------------
# RollingRate: what the worker reports as steps/s and episodes/s
# --------------------------------------------------------------------------

def test_rolling_rate_is_zero_before_any_work():
    assert RollingRate(60.0, clock=FakeClock()).rate() == 0.0


def test_rolling_rate_is_items_over_seconds_of_work():
    clock = FakeClock()
    r = RollingRate(60.0, clock=clock)
    r.add(4200, 12.0)
    assert r.rate() == pytest.approx(350.0)


def test_rolling_rate_aggregates_samples_not_averages_of_rates():
    # a fast 1s chunk and a slow 9s chunk must weight by time, not by count
    clock = FakeClock()
    r = RollingRate(60.0, clock=clock)
    r.add(1000, 1.0)
    clock.advance(1.0)
    r.add(900, 9.0)
    assert r.rate() == pytest.approx(190.0)  # 1900 items / 10s, not (1000+100)/2


def test_rolling_rate_forgets_samples_outside_the_window():
    clock = FakeClock()
    r = RollingRate(10.0, clock=clock)
    r.add(1000, 1.0)          # a cold, fast start
    clock.advance(30.0)       # ...half a minute of throttling later
    assert r.rate() == 0.0
    r.add(100, 1.0)
    assert r.rate() == pytest.approx(100.0)
    assert len(r) == 1


@pytest.mark.parametrize("items,seconds", [
    (100, 0.0),          # a zero-duration chunk would divide by zero
    (100, -3.0),         # a clock that jumped backwards
    (-50, 1.0),          # negative work
    (None, 1.0),
    (100, None),
    ("lots", 1.0),
    (100, "fast"),
])
def test_rolling_rate_drops_impossible_samples(items, seconds):
    clock = FakeClock()
    r = RollingRate(60.0, clock=clock)
    r.add(items, seconds)
    assert len(r) == 0 and r.rate() == 0.0


def test_rolling_rate_survives_a_garbage_window():
    assert RollingRate("wide", clock=FakeClock()).window == 120.0


# --------------------------------------------------------------------------
# Worker wiring: the flag and the wire contract that carry all of the above
# --------------------------------------------------------------------------

@pytest.mark.parametrize("flag,expected", [
    ("auto", None), ("AUTO", None), ("  auto  ", None), ("", None), (None, None),
    ("8", "8"), (8, 8), ("0", "0"), ("banana", "banana"),
])
def test_parse_procs_flag(flag, expected):
    from es.worker import parse_procs_flag
    # anything that is not "auto" goes to plan_procs untouched: it owns the
    # decision of what a bad --procs means, so an autostarted worker on a
    # student's laptop never dies on a typo
    assert parse_procs_flag(flag) == expected


def test_make_stats_matches_the_coordinator_contract():
    from es.worker import make_stats
    clock = FakeClock()
    steps, episodes = RollingRate(60.0, clock=clock), RollingRate(60.0, clock=clock)
    steps.add(4200, 12.0)
    episodes.add(16, 12.0)
    stats = make_stats(procs=8, host="omen-wsl", step_rate=steps, ep_rate=episodes)
    assert set(stats) == {"procs", "steps_per_s", "episodes_per_s", "host"}
    assert stats["procs"] == 8 and stats["host"] == "omen-wsl"
    assert stats["steps_per_s"] == pytest.approx(350.0)
    assert stats["episodes_per_s"] == pytest.approx(1.333, abs=1e-3)


def test_make_stats_is_json_serialisable():
    import json

    from es.worker import make_stats
    clock = FakeClock()
    stats = make_stats(procs=4, host="m4", step_rate=RollingRate(60.0, clock=clock),
                       ep_rate=RollingRate(60.0, clock=clock))
    assert json.loads(json.dumps(stats)) == stats  # it rides inside a /result POST


def test_worker_module_imports_without_an_emulator():
    # workers import this on machines where stable-retro may not be installed
    # yet (the env import is deliberately lazy, inside _make_env)
    from es import worker
    assert worker.RATE_WINDOW_S > 0
    assert hasattr(worker, "evaluate_member")
