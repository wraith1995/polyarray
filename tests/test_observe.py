"""`polyarray.observe` — the staged compile trace (fem Track 1 observability).

The contract the consumers (pointwise / savo / oracle) instrument against:

* `off` is genuinely free and records nothing; the ambient trace is never `None`;
* a stage's mass/degree is measured with the analysis polyarray already owns, and a bulk
  (deferred) node is counted as deferred, never forced;
* measurement never breaks a compile — a probe that raises yields a Measurement carrying the
  error, and the caller proceeds;
* `warn` fires on the absolute ceilings AND on a stage-over-stage mass jump (the blow-up
  fingerprint), while an ordinary compile stays silent;
* `dump` writes one numbered directory per stage, and `dump_dir_for` hands that same directory
  to pyab so the lowering artefacts land beside the symbolic snapshot that produced them.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

import polyarray as pa
from polyarray import observe


def _prov(name: str) -> pa.Provenance:
    return pa.Provenance("vertex", "x", (), name)


def _poly_program(power: int = 3, n: int = 3, name: str = "t") -> pa.Program:
    """A statement-free program whose outputs are degree-``power`` monomials in ``n`` atoms."""
    p = pa.Program(name, inputs=[pa.SymInput("V", (n,), _prov("V"))])
    cells = np.asarray(p.input("V").cells)
    out = np.empty(n, dtype=object)
    for i, c in enumerate(cells):
        acc = c
        for _ in range(power - 1):
            acc = acc * c
        out[i] = acc
    p.add_output("result", out)
    return p


# --- measurement ------------------------------------------------------------


def test_measure_program_reports_mass_cells_degree_and_provenance():
    m = observe.measure(_poly_program(power=3, n=3))
    assert m.kind == "program"
    assert m.symbolic_cells == 3 and m.n_output_cells == 3
    assert m.out_mass == 6            # x^3 as a rational: numerator + denominator terms
    assert m.degree == 3.0            # the atoms are scored 1 each, so x^3 -> 3
    assert m.prov_kinds == {"vertex": 3}
    assert not m.error


def test_degree_tracks_the_polynomial_degree():
    assert observe.measure(_poly_program(power=1)).degree == 1.0
    assert observe.measure(_poly_program(power=5)).degree == 5.0


def test_measure_tolerates_a_broken_object_without_raising():
    class Exploding:
        @property
        def program(self):
            raise RuntimeError("boom")

    m = observe.measure(Exploding())
    assert m.kind == "unmeasurable"
    assert "boom" in m.error


def test_measure_numeric_and_none_are_weightless():
    assert observe.measure(None).kind == "none"
    assert observe.measure(np.zeros((4, 4))).out_mass == 0
    assert observe.measure(np.zeros((4, 4))).n_output_cells == 16


def test_measure_does_not_force_a_bulk_node():
    """A deferred (bulk) output is counted, never materialised.

    Reading a bulk node's value is exactly the forcing this module exists to avoid — a tracer
    that materialises the thing it is measuring causes the blow-up it is meant to report.
    """
    p = pa.Program("bulk", inputs=[pa.SymInput("V", (3,), _prov("V"))])
    [big] = p.emit_stmt(pa.EinsumStmtOp(spec="i,i->i"), [p.input("V"), p.input("V")],
                        [pa.OutSpec("b", (3,))], note="bulk", bulk=True)
    p.add_output("result", big)
    m = observe.measure(p)
    assert m.kind == "program"
    assert m.symbolic_cells == 0       # the bulk output contributes no monomials
    assert m.n_defer == 1              # ...and is reported as deferred instead


def test_measure_sums_a_sequence_of_artefacts():
    parts = [_poly_program(power=2), _poly_program(power=4)]
    m = observe.measure(parts)
    assert m.out_mass == sum(observe.measure(p).out_mass for p in parts)
    assert m.degree == 4.0             # the max over the parts


# --- levels -----------------------------------------------------------------


def test_off_records_nothing_and_the_ambient_trace_is_never_none():
    assert observe.trace() is not None          # outside any block: the null trace
    assert observe.active() is False
    with observe.observe_compile("x", level=observe.Level.OFF) as tr:
        assert tr.stage("s", _poly_program()) is None
        with tr.phase("p") as box:
            box.append(_poly_program())
    assert tr.snapshots == []


def test_level_from_env(monkeypatch):
    monkeypatch.delenv("FEM_OBSERVE", raising=False)
    assert observe.level_from_env() is observe.Level.WARN      # the default
    for name, lv in [("off", observe.Level.OFF), ("info", observe.Level.INFO),
                     ("DEBUG", observe.Level.DEBUG), ("dump", observe.Level.DUMP)]:
        monkeypatch.setenv("FEM_OBSERVE", name)
        assert observe.level_from_env() is lv


def test_a_typo_in_the_env_var_warns_rather_than_silently_disabling(monkeypatch):
    monkeypatch.setenv("FEM_OBSERVE", "verbose")
    with pytest.warns(UserWarning, match="FEM_OBSERVE"):
        assert observe.level_from_env() is observe.Level.WARN


def test_the_ambient_trace_is_scoped_to_the_block():
    outer = observe.trace()
    with observe.observe_compile("inner", level=observe.Level.INFO) as tr:
        assert observe.trace() is tr
        assert observe.active() is True
    assert observe.trace() is outer


# --- recording --------------------------------------------------------------


def test_stages_are_sequenced_and_deltas_are_relative_to_the_previous_stage():
    # Mass counts MONOMIALS, so it grows with the number of cells, not with the exponent —
    # `x**8` is one term just as `x**2` is.  Degree is the axis that tracks the exponent.
    with observe.observe_compile("seq", level=observe.Level.INFO) as tr:
        tr.stage("small", _poly_program(power=2, n=3))
        tr.stage("big", _poly_program(power=8, n=300))
    a, b = tr.snapshots
    assert (a.seq, b.seq) == (1, 2)
    assert b.parent == a.seq
    assert b.m.mass > a.m.mass
    assert (a.m.degree, b.m.degree) == (2.0, 8.0)
    assert "big" in tr.report() and "small" in tr.report()


def test_phase_times_the_block_and_records_its_result():
    with observe.observe_compile("ph", level=observe.Level.INFO) as tr:
        with tr.phase("build", note="hello") as box:
            box.append(_poly_program(power=4))
    [snap] = tr.snapshots
    assert snap.stage == "build"
    assert snap.m.degree == 4.0
    assert snap.ctx == {"note": "hello"}
    assert snap.elapsed_s >= 0.0


def test_a_phase_that_raises_is_still_recorded():
    """The stage that blew up is precisely the one the trace must contain."""
    with pytest.raises(ValueError, match="kaboom"):
        with observe.observe_compile("boom", level=observe.Level.INFO) as tr:
            with tr.phase("doomed"):
                raise ValueError("kaboom")
    assert [s.stage for s in tr.snapshots] == ["doomed"]


def test_nested_phases_are_recorded_at_increasing_depth():
    with observe.observe_compile("nest", level=observe.Level.INFO) as tr:
        with tr.phase("outer") as outer:
            with tr.phase("inner") as inner:
                inner.append(_poly_program())
            outer.append(_poly_program())
    inner_s, outer_s = tr.snapshots
    assert (inner_s.stage, outer_s.stage) == ("inner", "outer")
    assert inner_s.depth > outer_s.depth


# --- roll-up of stages inside a loop ----------------------------------------


def test_a_stage_repeated_in_a_loop_rolls_up_into_one_row():
    """`single_compile` runs once per enumerated match — hundreds of times in one assembly.

    Without roll-up that is an 800-row table and 800 dump directories.
    """
    with observe.observe_compile("loop", level=observe.Level.INFO) as tr:
        for _ in range(50):
            tr.stage("sample", _poly_program(power=2, n=3))
    [snap] = tr.snapshots
    assert snap.count == 50
    assert snap.m.mass == observe.measure(_poly_program(power=2, n=3)).mass   # the max occurrence
    assert "meas" in tr.report()


def test_rollup_keeps_the_largest_occurrence_and_its_context():
    with observe.observe_compile("loop2", level=observe.Level.INFO) as tr:
        tr.stage("sample", _poly_program(power=2, n=3), which="small")
        tr.stage("sample", _poly_program(power=2, n=900), which="big")
        tr.stage("sample", _poly_program(power=2, n=3), which="small-again")
    [snap] = tr.snapshots
    assert snap.count == 3
    assert snap.ctx["which"] == "big"                 # the ctx of the occurrence that dominates
    assert snap.m.n_output_cells == 900


def test_repeated_occurrences_are_sampled_geometrically_below_debug():
    """Measuring is O(program size); the instrumented boundaries sit in hot loops.

    Measuring EVERY occurrence made the default level 3.3× slower than no instrumentation at all
    on argyris (37s → 122s), so below `debug` the occurrences are sampled 1, 2, 4, 8, …
    """
    with observe.observe_compile("sampled", level=observe.Level.INFO) as tr:
        for _ in range(100):
            tr.stage("bind-field", _poly_program(power=2, n=3))
    [snap] = tr.snapshots
    assert snap.count == 100
    assert snap.n_measured == 7          # occurrences 1,2,4,8,16,32,64
    assert snap.sampled is True
    assert "meas" in tr.report()


def test_debug_and_above_measure_every_occurrence():
    """There the user explicitly asked for the detail and accepted the cost."""
    with observe.observe_compile("full", level=observe.Level.DEBUG) as tr:
        for _ in range(20):
            tr.stage("bind-field", _poly_program(power=2, n=3))
    [snap] = tr.snapshots
    assert (snap.count, snap.n_measured) == (20, 20)
    assert snap.sampled is False


def test_small_loops_keep_full_fidelity():
    """Sampling only kicks in once a stage repeats — a handful of runs are all measured."""
    with observe.observe_compile("small", level=observe.Level.INFO) as tr:
        for _ in range(4):
            tr.stage("represent", _poly_program(power=2, n=3))
    [snap] = tr.snapshots
    assert (snap.count, snap.n_measured) == (4, 3)   # occurrences 1, 2, 4


def test_sampling_still_times_and_counts_every_occurrence():
    """An occurrence that is not measured must still be counted and its time attributed."""
    with observe.observe_compile("timed", level=observe.Level.INFO) as tr:
        for _ in range(20):
            with tr.phase("slow"):
                pass
    [snap] = tr.snapshots
    assert snap.count == 20
    assert snap.n_measured < snap.count
    assert snap.elapsed_s > 0.0


def test_a_sequence_off_one_program_carries_its_identity():
    """`bind_field` returns a JET — per-order arrays off ONE program — so the sequence branch
    propagates the program identity rather than dropping it."""
    p = _poly_program(power=2, n=5)
    assert observe.measure([p.input("V"), p.input("V")]).program_id is not None
    parts = [_poly_program(power=2, n=5), _poly_program(power=2, n=5)]
    assert observe.measure(parts).program_id is None


def test_rollup_is_keyed_by_depth_so_nesting_is_not_conflated():
    """A phase records its OWN row at the enclosing depth; stages recorded while it is open sit
    one level deeper.  So a same-named stage inside a phase is a distinct row from one outside."""
    with observe.observe_compile("loop3", level=observe.Level.INFO) as tr:
        tr.stage("x", _poly_program())                 # depth 0
        with tr.phase("outer"):
            tr.stage("x", _poly_program())             # depth 1 — a distinct row
    assert sorted(s.depth for s in tr.snapshots if s.stage == "x") == [0, 1]


def test_a_repeated_stage_logs_once_not_once_per_iteration(caplog):
    with caplog.at_level(logging.INFO, logger="fem.observe"):
        with observe.observe_compile("quiet-loop", level=observe.Level.INFO) as tr:
            for _ in range(30):
                tr.stage("sample", _poly_program(power=2, n=3))
    per_stage = [r for r in caplog.records if r.getMessage().startswith("[quiet-loop]")]
    assert len(per_stage) == 1                         # not 30
    assert tr.snapshots[0].count == 30


def test_a_repeated_stage_warns_once_not_once_per_iteration(monkeypatch):
    monkeypatch.setenv("FEM_OBSERVE_MASS_CEILING", "3")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with observe.observe_compile("warn-loop", level=observe.Level.WARN) as tr:
            for _ in range(30):
                tr.stage("heavy", _poly_program(power=3, n=3))
    assert len(caught) == 1
    assert tr.snapshots[0].count == 30


def test_rollup_dumps_one_directory_per_stage_not_per_iteration(tmp_path):
    with observe.observe_compile("d", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        for _ in range(40):
            tr.stage("sample", _poly_program(power=2, n=3))
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1 and dirs[0].name == "01-sample"
    assert "ran     40×" in (dirs[0] / "stage.txt").read_text()


# --- warnings ---------------------------------------------------------------


def test_an_ordinary_compile_warns_about_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # any warning becomes a failure
        with observe.observe_compile("quiet", level=observe.Level.WARN) as tr:
            tr.stage("modest", _poly_program(power=3))
    assert tr.snapshots


def test_an_oversized_stage_warns(monkeypatch):
    monkeypatch.setenv("FEM_OBSERVE_MASS_CEILING", "3")
    with pytest.warns(UserWarning, match="output mass"):
        with observe.observe_compile("fat", level=observe.Level.WARN) as tr:
            tr.stage("heavy", _poly_program(power=3))


def test_a_degree_ceiling_breach_warns(monkeypatch):
    monkeypatch.setenv("FEM_OBSERVE_DEGREE_CEILING", "2")
    with pytest.warns(UserWarning, match="degree"):
        with observe.observe_compile("deg", level=observe.Level.WARN) as tr:
            tr.stage("climbed", _poly_program(power=6))


def test_a_stage_over_stage_mass_jump_warns_and_is_named(monkeypatch):
    """The blow-up fingerprint: not "this stage is big" but "this stage MADE it big"."""
    monkeypatch.setenv("FEM_OBSERVE_MASS_CEILING", "10000000")   # absolute ceilings out of the way
    monkeypatch.setenv("FEM_OBSERVE_DEGREE_CEILING", "10000")
    with pytest.warns(UserWarning, match="jumped"):
        with observe.observe_compile("jump", level=observe.Level.WARN) as tr:
            tr.stage("before", _poly_program(power=2, n=2))         # mass 4
            tr.stage("after", _poly_program(power=2, n=40_000))     # mass 80_000
    jump = tr.jump_stage()
    assert jump is not None and jump.stage == "after"
    assert "after" in tr.report()


def test_warnings_are_silent_at_off():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with observe.observe_compile("mute", level=observe.Level.OFF) as tr:
            tr.stage("huge", _poly_program(power=9, n=200))
    assert tr.snapshots == []


# --- off-path detection -----------------------------------------------------


def test_off_path_warns_and_is_recorded_when_a_trace_is_active():
    """A trace that silently omits the route actually taken reads like a complete account of the
    compile. The whole value of this warning is "your trace has a hole here"."""
    with pytest.warns(UserWarning, match="OFF-PATH"):
        with observe.observe_compile("t", level=observe.Level.WARN) as tr:
            tr.stage("normal", _poly_program())
            observe.off_path("pointwise.integrate", "the Theme-A path")
    assert "pointwise.integrate" in tr.off_paths
    assert tr.off_paths["pointwise.integrate"] == "the Theme-A path"
    assert any(s.stage == "off-path/pointwise.integrate" for s in tr.snapshots)


def test_off_path_is_silent_when_nobody_is_observing():
    """Ordinary consumers of these APIs must see nothing — the warning is only meaningful to
    someone holding a trace."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        observe.off_path("pointwise.integrate", "not observing")     # no active trace
        with observe.observe_compile("t", level=observe.Level.OFF):
            observe.off_path("pointwise.integrate", "observing at OFF")


def test_off_path_warns_once_per_route_not_once_per_call():
    """It flags a route TAKEN, not a call count — a per-call warning would be a flood."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with observe.observe_compile("t", level=observe.Level.WARN) as tr:
            for _ in range(25):
                observe.off_path("pointwise.dumb_backend", "the cross-check oracle")
    assert len(caught) == 1
    assert len(tr.off_paths) == 1


def test_distinct_off_paths_are_reported_separately():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with observe.observe_compile("t", level=observe.Level.WARN) as tr:
            observe.off_path("a", "first")
            observe.off_path("b", "second")
    assert len(caught) == 2
    assert list(tr.off_paths) == ["a", "b"]


def test_the_report_flags_off_paths_so_the_hole_is_visible_in_the_table():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with observe.observe_compile("t", level=observe.Level.INFO) as tr:
            tr.stage("normal", _poly_program())
            observe.off_path("pointwise.integrate", "the Theme-A path")
    text = tr.report()
    assert "OFF-PATH" in text and "pointwise.integrate" in text and "the Theme-A path" in text


# --- report -----------------------------------------------------------------


def test_report_names_the_peak_and_the_jump():
    with observe.observe_compile("rep", level=observe.Level.INFO) as tr:
        tr.stage("a", _poly_program(power=2, n=2))
        tr.stage("b", _poly_program(power=2, n=40_000))
        tr.stage("c", _poly_program(power=2, n=40_000))
    text = tr.report()
    assert "peak:" in text and "largest jump:" in text
    assert tr.peak_stage().m.mass >= tr.snapshots[0].m.mass
    assert tr.jump_stage().stage == "b"          # b is where it GOT big; c merely stayed big


def test_report_survives_an_unmeasurable_stage():
    class Exploding:
        @property
        def program(self):
            raise RuntimeError("boom")

    with observe.observe_compile("bad", level=observe.Level.INFO) as tr:
        tr.stage("broken", Exploding())
    assert "broken" in tr.report()
    assert tr.snapshots[0].m.error


# --- dumps + the pyab/torch hand-off ---------------------------------------


def test_dump_writes_one_numbered_directory_per_stage(tmp_path):
    with observe.observe_compile("d", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        tr.stage("sample", _poly_program(power=2))
        tr.stage("lower", _poly_program(power=4))
    assert (tmp_path / "01-sample" / "stage.txt").exists()
    assert (tmp_path / "02-lower" / "stage.txt").exists()
    assert (tmp_path / "report.txt").exists()
    text = (tmp_path / "02-lower" / "stage.txt").read_text()
    assert "operand_mass" in text and "degree" in text and "IRReport" in text


def test_dump_dir_for_reserves_the_directory_the_stage_will_use(tmp_path):
    """The torch hand-off: `dump_dir_for(name)` before `stage(name, ...)` yields ONE directory
    holding both the symbolic snapshot and whatever pyab writes there."""
    with observe.observe_compile("h", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        tr.stage("sample", _poly_program())
        d = tr.dump_dir_for("value-kernel")             # what savo hands to compile_torch
        assert d == tmp_path / "02-value-kernel"
        (d / "f").mkdir()                               # stand in for pyab's per-function dir
        (d / "f" / "torch.py").write_text("# generated")
        tr.stage("value-kernel", _poly_program())
    assert (tmp_path / "02-value-kernel" / "stage.txt").exists()      # ours
    assert (tmp_path / "02-value-kernel" / "f" / "torch.py").exists() # pyab's, same directory


def test_dump_writes_the_polyarray_ir_as_readable_source(tmp_path):
    """"What is the output in polyarray" — the stage dir carries the actual program, not just
    its size, rendered with the renderer polyarray already owns."""
    with observe.observe_compile("src", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        tr.stage("represent", _poly_program(power=3, n=2))
    src = (tmp_path / "01-represent" / "program.py").read_text()
    assert "def " in src and "V" in src          # a real rendered function over the V atoms


def test_dump_renders_the_ir_from_a_sequence_of_artefacts(tmp_path):
    """savo stages a LIST of per-match integrands; rendering must not silently write nothing."""
    with observe.observe_compile("seq", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        tr.stage("represent-matches", [_poly_program(power=2), _poly_program(power=3)])
    src = (tmp_path / "01-represent-matches" / "program.py").read_text()
    assert "produced 2 programs" in src and "def " in src


def test_dump_writes_the_callers_own_detail(tmp_path):
    with observe.observe_compile("det", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        tr.stage("represent-matches", _poly_program(),
                 detail=lambda: "basis: Argyris\nterms: u*v\n")
    text = (tmp_path / "01-represent-matches" / "detail.txt").read_text()
    assert "basis: Argyris" in text and "terms: u*v" in text


def test_detail_is_a_thunk_evaluated_only_at_dump_and_only_once():
    """It may close over locals not yet bound when the phase is entered, and must not run at all
    below `dump` — a description can be expensive to build."""
    calls = []

    def describe():
        calls.append(1)
        return f"late-bound value = {value}"

    for lv in (observe.Level.OFF, observe.Level.WARN, observe.Level.INFO, observe.Level.DEBUG):
        with observe.observe_compile("t", level=lv) as tr:
            with tr.phase("s", detail=describe) as box:
                value = 42                          # bound INSIDE the phase
                box.append(_poly_program())
    assert calls == []                              # never called below `dump`


def test_detail_thunk_sees_late_bound_locals_and_is_not_re_run_per_occurrence(tmp_path):
    calls = []

    def describe():
        calls.append(1)
        return f"late-bound value = {value}"

    with observe.observe_compile("t", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        for _ in range(5):
            with tr.phase("s", detail=describe) as box:
                value = 42                          # assigned inside; read when the phase exits
                box.append(_poly_program())         # same size each time -> no new maximum
    assert (tmp_path / "01-s" / "detail.txt").read_text().strip() == "late-bound value = 42"
    assert calls == [1]                             # once, not once per occurrence


def test_detail_describes_the_SAME_occurrence_the_numbers_describe(tmp_path):
    """`stage.txt` reports the LARGEST occurrence, so `detail.txt` must too.

    Caching the first render would pair a description of occurrence 1 with the measurements of
    occurrence 900 — the two files would disagree about which run they are talking about.
    """
    sizes = [3, 5, 400, 4]
    seen: list[int] = []

    def describe():
        seen.append(current)
        return f"this occurrence had n={current}"

    with observe.observe_compile("t", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        for current in sizes:
            tr.stage("s", _poly_program(power=2, n=current), detail=describe)
    [snap] = tr.snapshots
    assert snap.m.n_output_cells == 400                       # the numbers track the largest
    assert "n=400" in (tmp_path / "01-s" / "detail.txt").read_text()   # ...and so does the detail
    assert seen[-1] == 400 and len(seen) < len(sizes)         # re-rendered only on a new maximum


def test_a_raising_detail_thunk_does_not_break_the_compile(tmp_path):
    def boom():
        raise RuntimeError("describe failed")

    with observe.observe_compile("t", level=observe.Level.DUMP, dump_root=tmp_path) as tr:
        snap = tr.stage("s", _poly_program(), detail=boom)
    assert snap is not None
    assert "describe failed" in (tmp_path / "01-s" / "detail.txt").read_text()


def test_dump_dir_for_is_none_below_dump_level(tmp_path):
    """So a call site can pass it straight through to `compile_torch(dump_dir=...)` unguarded."""
    for lv in (observe.Level.OFF, observe.Level.WARN, observe.Level.INFO, observe.Level.DEBUG):
        with observe.observe_compile("n", level=lv, dump_root=tmp_path) as tr:
            assert tr.dump_dir_for("k") is None
    assert observe.dump_dir_for("k") is None            # and on the null trace


def test_a_failing_dump_does_not_break_the_compile(tmp_path, caplog):
    tr = observe.CompileTrace("f", level=observe.Level.DUMP, dump_root=tmp_path / "root")
    (tmp_path / "root").write_text("not a directory")   # make mkdir under it fail
    with caplog.at_level(logging.WARNING, logger="fem.observe"):
        snap = tr.stage("doomed", _poly_program())
    assert snap is not None                             # the stage is still recorded
    assert any("dump" in r.message for r in caplog.records)
