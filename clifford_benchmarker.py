"""
benchmark.py
============
Compares CutQC wall-clock evaluation time between:
  - Baseline: clifford_weight=0.0 (original behaviour)
  - Modified: clifford_weight=<value> (Clifford-aware cutting)

Runs N_SEEDS trials per configuration and reports mean/std evaluate time,
number of cuts, and non-Clifford gate distribution across subcircuits.

Usage:
    python clifford_benchmarker.py
"""

import logging
import statistics
import math
from cutqc.main import CutQC
from helper_functions.benchmarks import generate_circ

from qiskit import QuantumCircuit
from qiskit.circuit.library import TGate, RXGate, RYGate, CZGate
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit

def make_symmetric_ladder(n_per_side=4):
    from qiskit import QuantumCircuit
    import numpy as np
    
    n = n_per_side * 2
    qc = QuantumCircuit(n)
    
    # Both sides have a mix of Clifford and non-Clifford
    # but non-Clifford gates are clustered toward the LEFT end of each side
    for q in range(n_per_side - 1):
        qc.cx(q, q + 1)
        if q < n_per_side // 2:
            qc.u3(np.pi/3, np.pi/5, np.pi/7, q)  # non-Clifford on left half
        else:
            qc.h(q)                                # Clifford on right half

    for q in range(n_per_side, n - 1):
        qc.cx(q, q + 1)
        if q < n_per_side + n_per_side // 2:
            qc.h(q)                                # Clifford on left half
        else:
            qc.u3(np.pi/3, np.pi/5, np.pi/7, q)  # non-Clifford on right half

    # Two bridges at asymmetric positions
    # Bridge 1 separates NC-heavy regions, bridge 2 mixes them
    qc.cx(n_per_side // 2 - 1, n_per_side)
    qc.cx(n_per_side - 1, n - 1)
    
    return qc

def make_near_clifford_vqe(num_qubits, depth, num_non_clifford=2):
    """
    Build a HWEA circuit where all but num_non_clifford rotation gates
    are set to Clifford angles (multiples of pi/2).
    """
    from helper_functions.benchmarks import gen_hwea
    circuit = gen_hwea(num_qubits, depth, regname="q")
    
    clifford_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    non_clifford_count = 0
    
    new_circ = circuit.copy()
    new_circ.clear()
    
    for instruction in circuit.data:
        gate = instruction.operation
        qargs = instruction.qubits
        if hasattr(gate, 'params') and len(gate.params) > 0:
            if non_clifford_count < num_non_clifford:
                # keep this one as non-Clifford
                non_clifford_count += 1
                new_circ.append(gate, qargs)
            else:
                # snap to nearest Clifford angle
                new_params = [min(clifford_angles, key=lambda a: abs(a - float(p))) 
                              for p in gate.params]
                new_gate = gate.__class__(*new_params)
                new_circ.append(new_gate, qargs)
        else:
            new_circ.append(gate, qargs)
    
    return new_circ

def make_multi_partition_circuit(n_per_block=6, n_bridges=2):
    n = n_per_block * 2
    qc = QuantumCircuit(n)
    
    # Block 0 — non-Clifford heavy
    for q in range(n_per_block - 1):
        qc.cz(q, q + 1)
        qc.t(q)
        qc.rx(np.pi / 2, q)
        qc.ry(np.pi / 2, q + 1)
    
    # Block 1 — Clifford heavy
    for q in range(n_per_block, n - 1):
        qc.cz(q, q + 1)
        qc.h(q)
        qc.h(q + 1)
    
    # Bridge gates — spread across multiple qubit pairs to create ambiguity
    for i in range(n_bridges):
        qc.cz(i * (n_per_block // n_bridges), n_per_block + i)
        qc.cz(i, n - 1 - i)  # additional cross connections at different positions
    
    return qc

# ── Benchmark configuration ────────────────────────────────────────────────
CIRCUIT_TYPE = "supremacy"
CIRCUIT_DEPTH = 1
CIRCUIT_SIZE = 16
N_SEEDS = 1
CLIFFORD_WEIGHT = 1.0        # weight to test against baseline 0.0

CUTTER_CONSTRAINTS_BASE = {
    "max_subcircuit_width": math.ceil(CIRCUIT_SIZE / 4 * 3),
    "max_subcircuit_cuts": 10,
    "subcircuit_size_imbalance": 2,
    "max_cuts": 10,
    "num_subcircuits": [2, 3],
}
# ──────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="clifford_benchmarker.log",
    filemode="w",
)

NON_CLIFFORD_GATES = {"t", "rx", "ry", "u3"}

def count_non_clifford_gates(subcircuit):
    return sum(1 for instr in subcircuit.data if instr.operation.name in NON_CLIFFORD_GATES)

def run_one(circuit, cutter_constraints, label):
    """Run a single CutQC trial. Returns a result dict."""
    cutqc = CutQC(
        circuit=circuit,
        cutter_constraints=cutter_constraints,
        verbose=True,
    )
    cutqc.cut()
    cutqc.evaluate(num_shots_fn=None)
    cutqc.build(mem_limit=32, recursion_depth=1)

    subcircuits = cutqc.cut_solution["subcircuits"]
    nc_counts = [count_non_clifford_gates(sc) for sc in subcircuits]

    return {
        "label": label,
        "num_cuts": cutqc.cut_solution["num_cuts"],
        "evaluate_time": cutqc.times["evaluate"],
        "num_subcircuits": len(subcircuits),
        "nc_per_subcircuit": nc_counts,
        "nc_imbalance": max(nc_counts) - min(nc_counts),
    }


def print_summary(label, results):
    times = [r["evaluate_time"] for r in results]
    cuts = [r["num_cuts"] for r in results]
    imbalances = [r["nc_imbalance"] for r in results]

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times) if len(times) > 1 else 0.0
    mean_cuts = statistics.mean(cuts)
    mean_imb = statistics.mean(imbalances)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Trials            : {len(results)}")
    print(f"  Evaluate time     : {mean_t:.3f}s ± {std_t:.3f}s")
    print(f"  Mean cuts         : {mean_cuts:.1f}")
    print(f"  Mean NC imbalance : {mean_imb:.1f}  (higher = more concentrated)")
    print(f"\n  Per-trial breakdown:")
    for r in results:
        nc_str = " vs ".join(str(n) for n in r["nc_per_subcircuit"])
        print(
            f"    cuts={r['num_cuts']}  eval={r['evaluate_time']:.3f}s  "
            f"NC gates per subcircuit: [{nc_str}]  imbalance={r['nc_imbalance']}"
        )


def main():
    print(f"\nBenchmark: {CIRCUIT_TYPE} circuit") #, {CIRCUIT_SIZE} qubits, depth={CIRCUIT_DEPTH}")
    # print(f"Seeds: {N_SEEDS}  |  clifford_weight tested: {CLIFFORD_WEIGHT}")
    print(f"clifford_weight tested: {CLIFFORD_WEIGHT}")

    baseline_results = []
    modified_results = []

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        # circuit = generate_circ(
        #     num_qubits=CIRCUIT_SIZE,
        #     depth=CIRCUIT_DEPTH,
        #     circuit_type=CIRCUIT_TYPE,
        #     reg_name="q",
        #     connected_only=True,
        #     seed=seed,
        # )
        # circuit = make_near_clifford_vqe(num_qubits=16, depth=2, num_non_clifford=2)
        # circuit = make_symmetric_ladder(n_per_side=5)
        circuit = make_multi_partition_circuit(n_per_block=6, n_bridges=5)
        print(set(node.op.name for node in circuit_to_dag(circuit).topological_op_nodes()))
        print(circuit.num_qubits, circuit.num_nonlocal_gates())

        # Baseline
        constraints_base = {**CUTTER_CONSTRAINTS_BASE, "clifford_weight": 0.0}
        r_base = run_one(circuit, constraints_base, "baseline")
        baseline_results.append(r_base)
        nc_str = " vs ".join(str(n) for n in r_base["nc_per_subcircuit"])
        print(f"  baseline : cuts={r_base['num_cuts']}  eval={r_base['evaluate_time']:.3f}s  NC=[{nc_str}]")

        # Modified
        constraints_mod = {**CUTTER_CONSTRAINTS_BASE, "clifford_weight": CLIFFORD_WEIGHT}
        r_mod = run_one(circuit, constraints_mod, f"clifford_weight={CLIFFORD_WEIGHT}")
        modified_results.append(r_mod)
        nc_str = " vs ".join(str(n) for n in r_mod["nc_per_subcircuit"])
        print(f"  modified : cuts={r_mod['num_cuts']}  eval={r_mod['evaluate_time']:.3f}s  NC=[{nc_str}]")

    print_summary("BASELINE  (clifford_weight=0.0)", baseline_results)
    print_summary(f"MODIFIED  (clifford_weight={CLIFFORD_WEIGHT})", modified_results)

    # Delta summary
    base_mean = statistics.mean(r["evaluate_time"] for r in baseline_results)
    mod_mean = statistics.mean(r["evaluate_time"] for r in modified_results)
    delta_pct = (mod_mean - base_mean) / base_mean * 100

    base_imb = statistics.mean(r["nc_imbalance"] for r in baseline_results)
    mod_imb = statistics.mean(r["nc_imbalance"] for r in modified_results)

    print(f"\n{'='*55}")
    print(f"  DELTA SUMMARY")
    print(f"{'='*55}")
    print(f"  Evaluate time change : {delta_pct:+.1f}%  ({'faster' if delta_pct < 0 else 'slower'})")
    print(f"  NC imbalance change  : {base_imb:.1f} → {mod_imb:.1f}  (baseline → modified)")
    print(f"  Verdict: ", end="")
    if delta_pct < -5 and mod_imb > base_imb:
        print("✓ Clifford-awareness helped: faster evaluation AND more concentrated NC gates.")
    elif mod_imb > base_imb and delta_pct >= -5:
        print("~ NC gates more concentrated but no significant speedup yet. Try increasing clifford_weight.")
    elif delta_pct < -5 and mod_imb <= base_imb:
        print("? Faster but NC concentration did not improve — speedup may be from fewer cuts.")
    else:
        print("✗ No clear benefit observed at this clifford_weight. Try a different value.")
    print()


if __name__ == "__main__":
    main()