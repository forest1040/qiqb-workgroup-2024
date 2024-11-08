from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts
from typing import Optional, TYPE_CHECKING, Iterable, Callable, TypeVar
from collections.abc import Iterable, Sequence
from quri_parts.circuit import ImmutableQuantumCircuit

from quri_parts.openqasm.circuit import convert_to_qasm_str
from quri_parts.riqu.backend import RiquSamplingBackend
from quri_parts.core.utils.concurrent import execute_concurrently

T_common = TypeVar("T_common")
T_individual = TypeVar("T_individual")
R = TypeVar("R")
Any = object()

if TYPE_CHECKING:
  from concurrent.futures import Executor

backend = RiquSamplingBackend()

def _sample(circuit: ImmutableQuantumCircuit, shots: int) -> MeasurementCounts:
  qasm = convert_to_qasm_str(circuit)
  job = backend.sample_qasm(qasm,n_shots=shots)
  result = job.result()
  return result

def _sample_sequentially(_:Any,circuit_shots_tuples:Iterable[tuple[ImmutableQuantumCircuit, int]]
) -> Iterable[MeasurementCounts]:
  return [_sample(circuit,shots) for circuit, shots in circuit_shots_tuples]

def _sample_concurrently(
    circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]],
    executor: Optional["Executor"],
    concurrency: int = 1
) -> Iterable[MeasurementCounts]:
  return execute_concurrently(
    _sample_sequentially,None,circuit_shots_tuples,executor,concurrency
  )


def create_riqu_concurrent_sampler(
    executor: Optional["Executor"] = None, concurrency: int = 1
) -> ConcurrentSampler:
  def sampler(
          circuit_shots_tuples: Iterable[tuple[ImmutableQuantumCircuit, int]]
  ) -> Iterable[MeasurementCounts]:
    return _sample_concurrently(circuit_shots_tuples,executor,concurrency)

  return sampler