################################################################################
# © Copyright 2026 Bench-Q Windows-support fork (terin678/benchq)
################################################################################
"""Regression tests for pyscf being an optional dependency.

The core resource estimation pipeline must work without pyscf installed
(pyscf does not build on native Windows), while pyscf-dependent modules
must fail with an actionable message pointing at the install extra.

These tests adapt to the environment: with pyscf installed (Linux/WSL full
install) they verify the modules import; without it (native Windows core
install) they verify the guard rails.
"""

import importlib
import importlib.util

import pytest

# The guarded modules need the full 'pyscf' install extra: pyscf itself plus
# openfermionpyscf (their import chains route through molecular Hamiltonian
# ingestion, which uses both).
PYSCF_EXTRA_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("pyscf", "openfermionpyscf")
)

CORE_MODULES = [
    "benchq.algorithms.data_structures",
    "benchq.compilation.graph_states",
    "benchq.compilation.graph_states.implementation_compiler",
    "benchq.logical_architecture_modeling.graph_based_logical_architectures",
    "benchq.quantum_hardware_modeling",
    "benchq.resource_estimators.graph_estimator",
]

PYSCF_GUARDED_MODULES = [
    "benchq.problem_embeddings.qpe",
    "benchq.problem_embeddings.block_encodings.double_factorized_hamiltonian",
    "benchq.problem_ingestion.molecular_hamiltonians._hamiltonian_generation",
]


# These tests assert importability, not warning hygiene; third-party imports
# (sympy/mpmath, qiskit) emit DeprecationWarnings that the repo-wide
# "warnings as errors" config would otherwise turn into false failures.
@pytest.mark.filterwarnings("default")
@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports_without_pyscf(module_name):
    """The estimation path must never acquire a hard pyscf dependency."""
    importlib.import_module(module_name)


@pytest.mark.filterwarnings("default")
@pytest.mark.parametrize("module_name", PYSCF_GUARDED_MODULES)
def test_pyscf_guarded_module(module_name):
    if PYSCF_EXTRA_AVAILABLE:
        importlib.import_module(module_name)
    else:
        with pytest.raises(ModuleNotFoundError, match=r"benchq\[pyscf\]"):
            importlib.import_module(module_name)


# Same star-topology GHZ as examples/data/ghz_circuit.qasm, so the golden
# values below correspond to ex_1_from_qasm.py output. (A chain-topology GHZ
# compiles to a shallower graph state: same qubit counts, 5.5e-5 s runtime.)
GHZ_4_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
cx q[0],q[1];
cx q[0],q[2];
cx q[0],q[3];
"""


@pytest.mark.slow
@pytest.mark.filterwarnings("default")
def test_ghz4_resource_estimate_regression():
    """Golden values for ex_1's GHZ-4 pipeline; identical on Linux and Windows.

    Verified 2026-06-10 on WSL Ubuntu and native Windows 11
    (CPython 3.11.13, Julia 1.12.6).
    """
    from qiskit.circuit import QuantumCircuit

    from benchq.algorithms.data_structures import (
        AlgorithmImplementation,
        ErrorBudget,
    )
    from benchq.compilation.graph_states import get_ruby_slippers_circuit_compiler
    from benchq.compilation.graph_states.implementation_compiler import (
        get_implementation_compiler,
    )
    from benchq.logical_architecture_modeling.graph_based_logical_architectures import (
        AllToAllArchitectureModel,
    )
    from benchq.quantum_hardware_modeling import BASIC_SC_ARCHITECTURE_MODEL
    from benchq.resource_estimators.graph_estimator import GraphResourceEstimator

    circuit = QuantumCircuit.from_qasm_str(GHZ_4_QASM)
    implementation = AlgorithmImplementation.from_circuit(
        circuit, ErrorBudget.from_even_split(total_failure_tolerance=1e-3), 1
    )
    estimator = GraphResourceEstimator(optimization="Time", verbose=False)
    compiler = get_implementation_compiler(
        circuit_compiler=get_ruby_slippers_circuit_compiler(),
        destination="single-thread",
    )
    info = estimator.compile_and_estimate(
        implementation,
        compiler,
        AllToAllArchitectureModel(),
        BASIC_SC_ARCHITECTURE_MODEL,
    )

    assert info.n_abstract_logical_qubits == 4
    assert info.n_physical_qubits == 3872
    assert info.n_t_gates == 0
    arch = info.logical_architecture_resource_info
    assert arch.num_logical_data_qubits == 16
    assert arch.data_and_bus_code_distance == 11
    assert arch.num_magic_state_factories == 0
    assert info.total_time_in_seconds == pytest.approx(7.7e-05)
    assert info.total_circuit_failure_rate == pytest.approx(4.35e-05, rel=1e-2)
