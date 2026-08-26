# resources.py -- how much of THIS machine an ES worker is allowed to take.
#
# Sizing is a pure function of (topology, flags): nothing here reads the host
# unless you let it. Every entry point takes the module it would otherwise
# import (psutil_mod, os_mod) so tests inject fakes instead of asserting
# against whatever box happens to run them -- the fleet is a 13900K, two
# 275HX laptops under WSL2 and an M4, and no single machine can stand in for
# the others.
#
# psutil is OPTIONAL everywhere. A worker box with only requirements-es.txt
# (numpy) installed still sizes itself from os.cpu_count(); psutil buys the
# physical-core count and the battery sensor, and its absence degrades to
# "unknown", never to an error.
#
# WHY procs map 1:1 to cores: libretro is single threaded and one emulator per
# OS process is a hard API limit (see worker.py), so one evaluation process
# saturates exactly one core for as long as it lives. There is no thread pool
# to tune here -- the only question this module answers is how many cores of
# this particular machine the fleet may eat.

import os
import sys
import time
from dataclasses import dataclass

__all__ = ["Topology", "detect_topology", "plan_procs", "apply_nice", "on_battery", "RollingRate"]

# psutil_mod / os_mod convention, used by every function below:
#   None   -> autodetect (lazy `import psutil`, or the real `os` module)
#   False  -> force the "psutil is not installed" path (tests use this, since
#             the test box may well have psutil for unrelated reasons)
#   module -> use it as given (a fake in tests, the real thing in production)
NO_PSUTIL = False


# ---------------------------------------------------------------------------
# Coercion helpers. Everything crossing this module's boundary comes from a
# CLI string, a fake, or a psutil that is entitled to return None on some
# platform, so nothing is trusted to be a number.
# ---------------------------------------------------------------------------

def _as_int(value):
    """int(value), or None when it is not a number. bool is rejected on
    purpose: `--procs True` is a bug, not a request for one process."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    """float(value), or None for non-numbers and NaN (NaN would poison every
    comparison downstream and silently pick a branch nobody intended)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _positive(value):
    """A strictly positive int, else None."""
    n = _as_int(value)
    return n if n is not None and n > 0 else None


def _discard(_message):
    """Default sink for warnings: plan_procs is pure and silent unless the
    caller hands it a logger (worker.py passes its own `[worker] ...` print)."""


def _call(mod, name, **kwargs):
    """mod.name(**kwargs) or None -- absent attribute or any raise degrades."""
    fn = getattr(mod, name, None)
    if fn is None:
        return None
    try:
        return fn(**kwargs)
    except Exception:  # noqa: BLE001 -- a probe that raises means "unknown"
        return None


def _resolve_psutil(psutil_mod):
    """See the NO_PSUTIL convention above."""
    if psutil_mod is None:
        try:
            import psutil  # optional dependency: absent on a minimal worker box
            return psutil
        except Exception:  # noqa: BLE001 -- broken install counts as absent
            return None
    return psutil_mod or None


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Topology:
    """What we could learn about this machine's CPUs.

    Never partially unknown: physical_cpus falls back to logical_cpus and
    logical_cpus to 1, so callers do arithmetic instead of None checks.
    physical_known says whether physical_cpus was measured or assumed -- it
    matters because without psutil a hyperthreaded box (the 13900K reports 32
    logical for 24 physical cores) would otherwise be sized 8 procs too fat.
    """
    logical_cpus: int
    physical_cpus: int
    platform: str
    physical_known: bool = True


def _affinity_count(os_mod):
    """Cores this PROCESS may actually use. On Linux/WSL this is the number
    that matters: a .wslconfig `processors=20`, a cgroup quota or a taskset
    makes cpu_count() describe a machine we are not allowed to fill."""
    fn = getattr(os_mod, "sched_getaffinity", None)
    if fn is None:
        return None
    try:
        return _positive(len(fn(0)))
    except Exception:  # noqa: BLE001
        return None


def _is_wsl(os_mod):
    env = getattr(os_mod, "environ", None) or {}
    try:
        if env.get("WSL_DISTRO_NAME") or env.get("WSL_INTEROP"):
            return True
    except Exception:  # noqa: BLE001 -- a fake environ need not be a mapping
        pass
    uname = _call(os_mod, "uname")
    release = getattr(uname, "release", "") if uname is not None else ""
    return "microsoft" in str(release).lower()


def _platform_name(os_mod):
    """Short label for the worker's startup line: darwin / linux / wsl / win32.
    sys.platform is not injectable, so a fake os module may carry the override
    attribute `sys_platform` -- that is the test seam for this one string."""
    override = getattr(os_mod, "sys_platform", None)
    base = str(override) if override else sys.platform
    if base.startswith("linux") and _is_wsl(os_mod):
        return "wsl"  # a WSL2 box is a Linux worker with a Windows owner watching
    return base


def detect_topology(psutil_mod=None, os_mod=None):
    """Best-effort CPU picture of this machine. Never raises, never returns
    zero: an undetectable machine degrades to a single-process worker, which
    is slow but correct, rather than taking the box down with it."""
    os_mod = os if os_mod is None else os_mod
    ps = _resolve_psutil(psutil_mod)

    logical = physical = None
    if ps is not None:
        logical = _positive(_call(ps, "cpu_count", logical=True))
        physical = _positive(_call(ps, "cpu_count", logical=False))
    if logical is None:
        logical = _positive(_call(os_mod, "cpu_count"))

    affinity = _affinity_count(os_mod)
    if affinity is not None:
        logical = affinity if logical is None else min(logical, affinity)
    if logical is None:
        logical = 1

    physical_known = physical is not None
    physical = logical if physical is None else min(physical, logical)
    return Topology(logical_cpus=logical, physical_cpus=physical,
                    platform=_platform_name(os_mod), physical_known=physical_known)


# ---------------------------------------------------------------------------
# The sizing policy
# ---------------------------------------------------------------------------

def _clamp(n, low, high):
    return max(low, min(n, high))


def _cap(n, cap, say):
    if cap is not None and n > cap:
        say(f"--max-procs {cap} caps the {n} processes this machine would have run")
        return cap
    return n


def plan_procs(topology, requested=None, reserve_cores=2, cpu_share=None,
               max_procs=None, warn=None):
    """How many emulator processes this machine should run. Pure.

    Precedence: explicit --procs  >  --cpu-share  >  physical - reserve_cores.

      requested   an explicit count wins over the auto policy -- the owner of
                  the machine knows things we cannot detect (a 20-core WSL
                  slice, a box dedicated to the fleet, a benchmark). It is
                  still floored at 1, still obeys max_procs, and is warned
                  about when it exceeds what the machine can host.
      cpu_share   0.0-1.0, "donate this fraction of the machine":
                  procs = round(share * physical).
      otherwise   procs = physical - reserve_cores.

    WHY the default reserves cores: one emulator process pins one core for the
    whole run, and these machines are not a datacenter -- they are somebody's
    daily driver (the M4 especially). Leaving reserve_cores free is what keeps
    a browser, an IDE and a video call responsive while the box contributes.
    Because emulation is CPU-bound and single-threaded per instance, procs is
    exactly "cores taken", which is what makes this arithmetic and not a guess.

    `warn` is an optional callable taking one string; without it the function
    is silent (it stays a pure sizing decision that tests can call in a loop).
    """
    say = warn if callable(warn) else _discard
    physical = _positive(getattr(topology, "physical_cpus", None)) or 1

    cap = _positive(max_procs)
    if max_procs is not None and cap is None:
        say(f"--max-procs {max_procs!r} is not a positive integer; ignoring it")

    n = _as_int(requested)
    if requested is not None and n is None:
        say(f"--procs {requested!r} is not a number; sizing this machine automatically")
    if n is not None:
        # Sanity ceiling: honoring an oversubscribed --procs is deliberate
        # (contention, warned below), but a fat-fingered 999 in an autostarted
        # unit is an OOM on somebody's daily driver, not a choice. 4x physical
        # comfortably covers every legitimate oversubscription experiment.
        ceiling = 4 * physical
        if n > ceiling:
            say(f"--procs {n} is beyond any sane oversubscription for {physical} "
                f"cores; clamping to {ceiling}")
            n = ceiling
        if n > physical:
            say(f"--procs {n} exceeds this machine's {physical} cores: the emulators will "
                f"contend and per-process fps will drop (total throughput will not rise)")
        if n < 1:
            say(f"--procs {n} is below 1; running a single emulator process")
            n = 1
        return _cap(n, cap, say)

    # every branch from here on divides up the physical core count, so this is
    # where an assumed (psutil-less) count actually costs someone throughput
    if not getattr(topology, "physical_known", True):
        say(f"psutil not available: sizing from {physical} logical CPUs, which "
            f"over-counts a hyperthreaded machine -- `pip install psutil` or pass --procs N")

    share = _as_float(cpu_share)
    if cpu_share is not None and share is None:
        say(f"--cpu-share {cpu_share!r} is not a number; falling back to reserved-core sizing")
    if share is not None:
        if share <= 0:
            say(f"--cpu-share {share} donates nothing; running a single emulator process")
            share = 0.0
        elif share > 1.0:
            say(f"--cpu-share {share} is above 1.0; using the whole machine")
            share = 1.0
        # half-up, not round(): round(2.5) is 2 in Python, and a share that
        # lands exactly between two counts should round toward contributing.
        return _cap(_clamp(int(share * physical + 0.5), 1, physical), cap, say)

    reserve = _as_int(reserve_cores)
    if reserve is None or reserve < 0:
        say(f"--reserve-cores {reserve_cores!r} is not a count >= 0; reserving nothing")
        reserve = 0
    if reserve >= physical:
        say(f"reserving {reserve} of {physical} cores leaves no room to work; "
            f"running a single emulator process")
    return _cap(_clamp(physical - reserve, 1, physical), cap, say)


# ---------------------------------------------------------------------------
# Staying usable while contributing
# ---------------------------------------------------------------------------

def apply_nice(delta, os_mod=None, warn=None):
    """Raise this process's niceness by `delta`. Returns the resulting
    niceness, or None when the platform would not do it. Never raises.

    WHY this matters more than a low proc count: niceness is the only knob
    that yields cores ON DEMAND. At nice 10 the scheduler hands the CPU to
    anything interactive the moment it wants it -- a keystroke, a scroll, a
    compile -- and gives it back to the emulators microseconds later, so the
    machine can run at full procs and still feel idle. A hard proc cap cannot
    do that: it decides in advance to leave cores idle even when nobody wants
    them, which costs throughput all day to buy responsiveness for the few
    minutes the owner is actually typing. Use the cap for RAM/thermals and
    politeness; use nice for interactivity.

    Failure is a warning, never fatal: os.nice does not exist on native
    Windows, and lowering niceness (a negative delta) needs root on POSIX.
    """
    os_mod = os if os_mod is None else os_mod
    say = warn if callable(warn) else _discard
    n = _as_int(delta)
    if n is None:
        say(f"--nice {delta!r} is not an integer; leaving scheduling priority alone")
        return None
    fn = getattr(os_mod, "nice", None)
    if fn is None:
        say("os.nice is unavailable on this platform; worker priority left at default")
        return None
    try:
        return _as_int(fn(n))
    except Exception as e:  # noqa: BLE001 -- PermissionError, OSError, anything
        say(f"could not renice by {n} ({e}); worker priority left at default")
        return None


def on_battery(psutil_mod=None):
    """True on battery, False on wall power, None when unknown (no psutil, no
    sensor, or a desktop with no battery at all). Callers WARN on True and
    never block: a laptop on battery throttles hard and its fitnesses come in
    late, but it is still contributing, and the queue re-leases whatever it
    drops when it sleeps."""
    ps = _resolve_psutil(psutil_mod)
    if ps is None:
        return None
    battery = _call(ps, "sensors_battery")
    if battery is None:
        return None
    plugged = getattr(battery, "power_plugged", None)
    return None if plugged is None else not bool(plugged)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

class RollingRate:
    """Throughput over a rolling wall-clock window, as items per second of work.

    Samples are (items, seconds_of_work) pairs rather than timestamps of
    single events: a worker reports a whole chunk at once ("4180 agent steps
    in 12.3s across 8 procs"), and items/seconds is the honest aggregate for
    the machine. Batch the adds -- one per chunk, not one per emulator step.

    Old samples fall out of the window so a machine that starts throttling, or
    whose owner takes their cores back, shows up in /status within the window
    instead of being averaged against its first cold minutes forever.
    """

    def __init__(self, window_seconds=120.0, clock=time.monotonic):
        self.window = _as_float(window_seconds) or 120.0
        self._clock = clock
        self._samples = []  # (timestamp, items, seconds_of_work)

    def add(self, items, seconds, now=None):
        now = self._clock() if now is None else now
        items = _as_float(items)
        seconds = _as_float(seconds)
        # a zero-duration sample would divide by zero and a negative one is a
        # clock that jumped: both are dropped rather than poisoning the rate
        if items is not None and seconds is not None and seconds > 0 and items >= 0:
            self._samples.append((now, items, seconds))
        self._prune(now)

    def _prune(self, now):
        cutoff = now - self.window
        self._samples = [s for s in self._samples if s[0] >= cutoff]

    def rate(self, now=None):
        now = self._clock() if now is None else now
        self._prune(now)
        work = sum(s[2] for s in self._samples)
        return sum(s[1] for s in self._samples) / work if work > 0 else 0.0

    def __len__(self):
        return len(self._samples)
