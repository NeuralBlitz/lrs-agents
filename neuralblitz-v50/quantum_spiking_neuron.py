"""
Quantum Spiking Neuron - Production Grade Implementation
Architecture 1: Biological-Quantum Hybrid Neural Network Component

Mathematical Framework:
----------------------
This implementation combines the leaky integrate-and-fire neuron model with
quantum mechanical superposition states. The neuron exists in a 2-dimensional
Hilbert space H = C² with basis {|0⟩, |1⟩} representing quiescent and active
states respectively.

The time evolution follows the Schrödinger equation:
    iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩

Where the Hamiltonian Ĥ encodes the classical membrane potential:
    Ĥ = V(t) σz + Δ σx

With σz, σx being Pauli matrices and Δ the quantum tunneling amplitude.

The classical membrane potential follows:
    τ_m dV/dt = -(V - V_rest) + R·I(t)

Upon reaching threshold V_th, the quantum state collapses to |1⟩ (spike),
then resets via quantum Zeno effect to |0⟩.

References:
- Maass & Bishop (2001) Pulsed Neural Networks
- Nielsen & Chuang (2010) Quantum Computation and Quantum Information
- Schuld & Petruccione (2018) Supervised Learning with Quantum Computers
"""

from __future__ import annotations

# Ensure custom package paths are available for IDE/LSP
import sys

sys.path.insert(0, "/home/runner/workspace/NB-Ecosystem/lib/python3.11/site-packages")

import numpy as np  # type: ignore
from numpy.typing import NDArray  # type: ignore
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging
import time
from contextlib import contextmanager
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumSpikingError(Exception):
    """Base exception for quantum spiking neuron errors."""

    pass


class InvalidQuantumStateError(QuantumSpikingError):
    """Raised when quantum state violates normalization or other constraints."""

    pass


class NumericalInstabilityError(QuantumSpikingError):
    """Raised when numerical computations become unstable."""

    pass


class NeuronStateError(QuantumSpikingError):
    """Raised when neuron state is inconsistent or invalid."""

    pass


class IntegrationError(QuantumSpikingError):
    """Raised when temporal integration fails to converge."""

    pass


@dataclass(frozen=True)
class NeuronConfiguration:
    """Immutable configuration for quantum spiking neuron.

    Attributes:
        resting_potential: Resting membrane potential in mV (default: -70.0)
        threshold_potential: Spike threshold in mV (default: -55.0)
        membrane_time_constant: Membrane time constant τ_m in ms (default: 20.0)
        membrane_resistance: Membrane resistance R in MΩ (default: 1.0)
        refractory_period: Refractory period after spike in ms (default: 2.0)
        quantum_tunneling: Quantum tunneling amplitude Δ (default: 0.1)
        coherence_time: Quantum coherence time T₂ in ms (default: 100.0)
        dt: Integration time step in ms (default: 0.1)
        max_history_length: Maximum spike history to retain (default: 10000)
        numerical_tolerance: Tolerance for numerical comparisons (default: 1e-10)
    """

    resting_potential: float = -70.0
    threshold_potential: float = -55.0
    membrane_time_constant: float = 20.0
    membrane_resistance: float = 1.0
    refractory_period: float = 2.0
    quantum_tunneling: float = 0.1
    coherence_time: float = 100.0
    dt: float = 0.1
    max_history_length: int = 10000
    numerical_tolerance: float = 1e-10

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.membrane_time_constant <= 0:
            raise ValueError(
                f"Membrane time constant must be positive, got {self.membrane_time_constant}"
            )
        if self.dt <= 0 or self.dt > self.membrane_time_constant:
            raise ValueError(
                f"Integration step dt must be positive and less than τ_m, got {self.dt}"
            )
        if self.threshold_potential <= self.resting_potential:
            raise ValueError(f"Threshold must be greater than resting potential")
        if self.coherence_time <= 0:
            raise ValueError(f"Coherence time must be positive")
        if self.quantum_tunneling < 0:
            raise ValueError(f"Quantum tunneling amplitude must be non-negative")


@dataclass
class SpikeEvent:
    """Represents a spike event with temporal and quantum information.

    Attributes:
        timestamp: Time of spike occurrence in ms
        membrane_potential: Membrane potential at spike time
        quantum_state: Quantum state |ψ⟩ immediately after spike
        coherence: Quantum coherence C = |⟨ψ|ψ⟩|² at spike time
        spike_id: Unique identifier for this spike
    """

    timestamp: float
    membrane_potential: float
    quantum_state: NDArray[np.complex128]
    coherence: float
    spike_id: int = field(default_factory=lambda: int(time.time() * 1e6))

    def __post_init__(self):
        """Validate spike event data."""
        if len(self.quantum_state) != 2:
            raise InvalidQuantumStateError(
                f"Quantum state must be 2D, got {len(self.quantum_state)}D"
            )
        norm = np.linalg.norm(self.quantum_state)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise InvalidQuantumStateError(
                f"Quantum state must be normalized, got norm={norm}"
            )


@dataclass
class QuantumState:
    """Represents a quantum state in the 2D Hilbert space.

    The state |ψ⟩ = α|0⟩ + β|1⟩ is stored as a complex vector [α, β].
    Maintains normalization constraint |α|² + |β|² = 1.

    Attributes:
        amplitudes: Complex amplitudes [α, β]
        basis: Basis states description
    """

    amplitudes: NDArray[np.complex128]
    basis: Tuple[str, str] = ("|0⟩ (quiescent)", "|1⟩ (active)")

    def __post_init__(self):
        """Ensure proper initialization and normalization."""
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        if self.amplitudes.shape != (2,):
            raise InvalidQuantumStateError(
                f"State must be 2D, got shape {self.amplitudes.shape}"
            )
        self._normalize()

    def _normalize(self) -> None:
        """Normalize the quantum state."""
        norm = np.linalg.norm(self.amplitudes)
        if norm < 1e-15:
            raise InvalidQuantumStateError("Cannot normalize zero vector")
        self.amplitudes /= norm

    @property
    def alpha(self) -> complex:
        """Amplitude of |0⟩ state."""
        return self.amplitudes[0]

    @property
    def beta(self) -> complex:
        """Amplitude of |1⟩ state."""
        return self.amplitudes[1]

    @property
    def probability_ground(self) -> float:
        """Probability of measuring |0⟩: P(0) = |α|²."""
        return np.abs(self.alpha) ** 2

    @property
    def probability_excited(self) -> float:
        """Probability of measuring |1⟩: P(1) = |β|²."""
        return np.abs(self.beta) ** 2

    @property
    def coherence(self) -> float:
        """Quantum coherence measure C = |α* β|."""
        return np.abs(np.conj(self.alpha) * self.beta)

    def density_matrix(self) -> NDArray[np.complex128]:
        """Compute density matrix ρ = |ψ⟩⟨ψ|."""
        return np.outer(self.amplitudes, np.conj(self.amplitudes))

    def von_neumann_entropy(self) -> float:
        """Compute von Neumann entropy S = -Tr(ρ log ρ)."""
        rho = self.density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove zero eigenvalues
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def apply_unitary(self, U: NDArray[np.complex128]) -> QuantumState:
        """Apply unitary transformation U to state.

        Args:
            U: 2×2 unitary matrix

        Returns:
            New QuantumState after transformation

        Raises:
            InvalidQuantumStateError: If U is not unitary
        """
        U = np.asarray(U, dtype=np.complex128)
        if U.shape != (2, 2):
            raise InvalidQuantumStateError(f"Unitary must be 2×2, got {U.shape}")

        # Check unitarity: U†U = I
        if not np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10):
            raise InvalidQuantumStateError("Matrix is not unitary")

        new_amplitudes = U @ self.amplitudes
        return QuantumState(new_amplitudes, self.basis)

    def measure(self, basis: str = "computational") -> Tuple[int, QuantumState]:
        """Perform quantum measurement.

        Args:
            basis: Measurement basis ("computational" or "hadamard")

        Returns:
            Tuple of (outcome, collapsed_state)

        Raises:
            InvalidQuantumStateError: If basis is unknown
        """
        if basis == "computational":
            probabilities = [self.probability_ground, self.probability_excited]
            outcome = np.random.choice([0, 1], p=probabilities)
            new_state = QuantumState(
                np.array([1.0, 0.0]) if outcome == 0 else np.array([0.0, 1.0]),
                self.basis,
            )
        elif basis == "hadamard":
            # Transform to Hadamard basis
            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            transformed = H @ self.amplitudes
            probabilities = [np.abs(transformed[0]) ** 2, np.abs(transformed[1]) ** 2]
            outcome = np.random.choice([0, 1], p=probabilities)
            new_state = QuantumState(
                H.T @ (np.array([1.0, 0.0]) if outcome == 0 else np.array([0.0, 1.0])),
                self.basis,
            )
        else:
            raise InvalidQuantumStateError(f"Unknown basis: {basis}")

        return outcome, new_state

    def __repr__(self) -> str:
        return (
            f"QuantumState(|ψ⟩ = {self.alpha:.4f}|0⟩ + {self.beta:.4f}|1⟩, "
            f"C={self.coherence:.4f})"
        )


class QuantumSpikingNeuron:
    """Production-grade quantum spiking neuron implementation.

    This class implements a biologically-inspired spiking neuron with quantum
    mechanical superposition of active and quiescent states. The neuron integrates
    classical synaptic inputs while maintaining quantum coherence.

    Mathematical Model:
    ------------------
    Classical membrane dynamics:
        τ_m dV/dt = -(V - V_rest) + R·I_syn(t)

    Quantum state evolution:
        |ψ(t+dt)⟩ = exp(-iĤdt/ℏ)|ψ(t)⟩

    where Ĥ = V(t)σ_z + Δσ_x incorporates membrane potential and tunneling.

    Spike generation:
        If V(t) ≥ V_th: collapse to |1⟩, record spike, reset via Zeno effect

    Example:
        >>> config = NeuronConfiguration(
        ...     resting_potential=-70.0,
        ...     threshold_potential=-55.0,
        ...     membrane_time_constant=20.0
        ... )
        >>> neuron = QuantumSpikingNeuron("neuron_001", config)
        >>>
        >>> # Simulate with random inputs
        >>> for t in range(1000):
        ...     input_current = np.random.randn() * 5.0  # nA
        ...     spike, state = neuron.step(input_current, dt=0.1)
        ...     if spike:
        ...         print(f"Spike at t={t*0.1:.1f}ms!")

    Attributes:
        neuron_id: Unique identifier for this neuron
        config: Immutable configuration parameters
        membrane_potential: Current membrane potential V(t)
        quantum_state: Current quantum state |ψ(t)⟩
        spike_history: List of recorded SpikeEvent objects
        time_elapsed: Total simulation time elapsed
        is_refractory: Whether neuron is in refractory period
    """

    def __init__(
        self,
        neuron_id: str,
        config: Optional[NeuronConfiguration] = None,
        initial_state: Optional[QuantumState] = None,
    ):
        """Initialize quantum spiking neuron.

        Args:
            neuron_id: Unique identifier for this neuron
            config: Configuration parameters (uses defaults if None)
            initial_state: Initial quantum state (|0⟩ if None)

        Raises:
            ValueError: If neuron_id is empty or invalid
            TypeError: If config is not NeuronConfiguration instance
        """
        if not isinstance(neuron_id, str) or not neuron_id.strip():
            raise ValueError("neuron_id must be non-empty string")

        self.neuron_id = neuron_id
        self.config = config or NeuronConfiguration()

        if not isinstance(self.config, NeuronConfiguration):
            raise TypeError("config must be NeuronConfiguration instance")

        # State variables
        self._membrane_potential = self.config.resting_potential
        self._quantum_state = initial_state or QuantumState(np.array([1.0, 0.0]))
        self._spike_history: List[SpikeEvent] = []
        self._time_elapsed = 0.0
        self._is_refractory = False
        self._refractory_end_time = 0.0
        self._spike_count = 0

        # Precompute Pauli matrices for Hamiltonian
        self._sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Statistics
        self._computation_time = 0.0
        self._integration_steps = 0

        logger.info(f"Initialized QuantumSpikingNeuron {neuron_id}")

    @property
    def membrane_potential(self) -> float:
        """Current membrane potential in mV."""
        return self._membrane_potential

    @property
    def quantum_state(self) -> QuantumState:
        """Current quantum state."""
        return self._quantum_state

    @property
    def spike_history(self) -> Tuple[SpikeEvent, ...]:
        """Tuple of recorded spike events (immutable view)."""
        return tuple(self._spike_history)

    @property
    def time_elapsed(self) -> float:
        """Total simulation time elapsed in ms."""
        return self._time_elapsed

    @property
    def is_refractory(self) -> bool:
        """Whether neuron is currently in refractory period."""
        return self._is_refractory

    @property
    def spike_count(self) -> int:
        """Total number of spikes generated."""
        return self._spike_count

    @property
    def spike_rate(self) -> float:
        """Average spike rate in Hz over simulation time."""
        if self._time_elapsed < 1e-6:
            return 0.0
        return self._spike_count / (self._time_elapsed / 1000.0)

    def _compute_hamiltonian(self) -> NDArray[np.complex128]:
        """Compute Hamiltonian matrix Ĥ = V(t)σz + Δσx.

        Returns:
            2×2 Hermitian matrix representing the Hamiltonian
        """
        # Normalize membrane potential to dimensionless units
        V_norm = (self._membrane_potential - self.config.resting_potential) / (
            self.config.threshold_potential - self.config.resting_potential
        )

        H = V_norm * self._sigma_z + self.config.quantum_tunneling * self._sigma_x
        return H.astype(np.complex128)

    def _evolve_quantum_state(self, dt: float) -> None:
        """Evolve quantum state via Schrödinger equation.

        Uses matrix exponential: |ψ(t+dt)⟩ = exp(-iĤdt)|ψ(t)⟩

        Args:
            dt: Time step in ms

        Raises:
            NumericalInstabilityError: If evolution becomes numerically unstable
        """
        H = self._compute_hamiltonian()

        # Compute unitary: U = exp(-iHdt)
        # For small dt, use Padé approximation or direct matrix exponential
        try:
            from scipy.linalg import expm  # type: ignore

            U = expm(-1j * H * dt)
        except ImportError:
            # Fallback: Taylor series for matrix exponential (less accurate)
            U = np.eye(2, dtype=np.complex128)
            term = np.eye(2, dtype=np.complex128)
            for n in range(1, 10):
                term = term @ (-1j * H * dt) / n
                U += term

        # Apply unitary evolution
        new_amplitudes = U @ self._quantum_state.amplitudes

        # Check numerical stability
        if np.any(np.isnan(new_amplitudes)) or np.any(np.isinf(new_amplitudes)):
            raise NumericalInstabilityError(
                "Quantum state evolution produced NaN or Inf"
            )

        self._quantum_state = QuantumState(new_amplitudes)

    def _integrate_membrane(self, input_current: float, dt: float) -> None:
        """Integrate classical membrane equation.

        Uses exponential Euler method for numerical stability:
        V(t+dt) = V_rest + (V(t) - V_rest)exp(-dt/τ_m) + R·I·(1-exp(-dt/τ_m))

        Args:
            input_current: Input current I(t) in nA
            dt: Time step in ms
        """
        if self._is_refractory:
            # During refractory period, clamp at resting potential
            self._membrane_potential = self.config.resting_potential
            return

        tau = self.config.membrane_time_constant
        R = self.config.membrane_resistance
        V_rest = self.config.resting_potential

        # Exponential integration (exact solution for constant input)
        decay = np.exp(-dt / tau)
        self._membrane_potential = (
            V_rest
            + (self._membrane_potential - V_rest) * decay
            + R * input_current * (1 - decay)
        )

    def _check_spike(self) -> bool:
        """Check if membrane potential exceeds threshold.

        Returns:
            True if spike should be generated
        """
        return self._membrane_potential >= self.config.threshold_potential

    def _generate_spike(self) -> SpikeEvent:
        """Generate spike event with quantum collapse.

        When spike occurs:
        1. Collapse quantum state to |1⟩ (active)
        2. Record spike event
        3. Reset membrane potential
        4. Enter refractory period

        Returns:
            SpikeEvent with spike details
        """
        # Collapse quantum state to |1⟩
        collapsed_state = QuantumState(np.array([0.0, 1.0]))

        # Create spike event
        spike = SpikeEvent(
            timestamp=self._time_elapsed,
            membrane_potential=self._membrane_potential,
            quantum_state=collapsed_state.amplitudes.copy(),
            coherence=self._quantum_state.coherence,
        )

        # Record spike
        self._spike_history.append(spike)
        if len(self._spike_history) > self.config.max_history_length:
            self._spike_history.pop(0)

        # Reset membrane potential
        self._membrane_potential = self.config.resting_potential

        # Reset quantum state (quantum Zeno effect)
        self._quantum_state = QuantumState(np.array([1.0, 0.0]))

        # Enter refractory period
        self._is_refractory = True
        self._refractory_end_time = self._time_elapsed + self.config.refractory_period

        self._spike_count += 1

        logger.debug(f"Neuron {self.neuron_id} spiked at t={self._time_elapsed:.2f}ms")

        return spike

    def _update_refractory(self) -> None:
        """Update refractory state based on current time."""
        if self._is_refractory and self._time_elapsed >= self._refractory_end_time:
            self._is_refractory = False

    def step(
        self, input_current: Union[float, np.ndarray], dt: Optional[float] = None
    ) -> Tuple[bool, QuantumState]:
        """Perform single integration step.

        This is the primary method for evolving the neuron forward in time.
        It integrates both classical membrane dynamics and quantum state evolution.

        Args:
            input_current: Synaptic input current I(t) in nA. Can be scalar or
                array (in which case sum is taken).
            dt: Integration time step in ms (uses config.dt if None)

        Returns:
            Tuple of (did_spike, quantum_state_after_step)

        Raises:
            IntegrationError: If integration fails
            TypeError: If input_current has invalid type

        Example:
            >>> neuron = QuantumSpikingNeuron("test")
            >>> for _ in range(100):
            ...     spike, state = neuron.step(5.0)  # 5 nA input
            ...     if spike:
            ...         print("Spike!")
        """
        start_time = time.perf_counter()

        try:
            # Validate and process input
            if isinstance(input_current, np.ndarray):
                input_current = float(np.sum(input_current))
            else:
                input_current = float(input_current)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"input_current must be numeric, got {type(input_current)}: {e}"
            )

        dt = dt or self.config.dt

        try:
            # Update refractory state
            self._update_refractory()

            # Integrate classical dynamics
            self._integrate_membrane(input_current, dt)

            # Evolve quantum state
            self._evolve_quantum_state(dt)

            # Check for spike
            did_spike = False
            if self._check_spike() and not self._is_refractory:
                self._generate_spike()
                did_spike = True

            # Update time
            self._time_elapsed += dt
            self._integration_steps += 1

            # Apply decoherence
            self._apply_decoherence(dt)

        except Exception as e:
            raise IntegrationError(
                f"Integration failed at t={self._time_elapsed}: {e}"
            ) from e

        self._computation_time += time.perf_counter() - start_time

        return did_spike, self._quantum_state

    def _apply_decoherence(self, dt: float) -> None:
        """Apply environmental decoherence to quantum state.

        Decoherence reduces off-diagonal elements of density matrix:
        ρ(t+dt) = ρ(t) * exp(-dt/T₂)

        Args:
            dt: Time step in ms
        """
        if self.config.coherence_time == np.inf:
            return

        # Decoherence factor
        decay = np.exp(-dt / self.config.coherence_time)

        # Reduce coherence
        rho = self._quantum_state.density_matrix()
        # Off-diagonal decay
        rho[0, 1] *= decay
        rho[1, 0] *= decay

        # Reconstruct state (approximate - for full fidelity use Kraus operators)
        # This is a simplified model assuming pure states
        coherence = np.abs(rho[0, 1])
        if coherence > self.config.numerical_tolerance:
            # Maintain normalization
            p0 = np.real(rho[0, 0])
            p1 = np.real(rho[1, 1])

            # Reconstruct approximate pure state
            phase = np.angle(rho[0, 1])
            new_alpha = np.sqrt(p0)
            new_beta = np.sqrt(p1) * np.exp(1j * phase)

            self._quantum_state = QuantumState(np.array([new_alpha, new_beta]))

    def simulate(
        self, input_currents: NDArray[np.float64], dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run complete simulation with input current trace.

        Args:
            input_currents: Array of input currents in nA
            dt: Time step in ms (uses config.dt if None)

        Returns:
            Dictionary containing:
                - spike_times: Times of spikes in ms
                - spike_count: Total number of spikes
                - spike_rate: Average spike rate in Hz
                - membrane_trace: Array of membrane potentials
                - quantum_trace: Array of quantum states
                - final_state: Final quantum state
                - simulation_time: Total simulation time in ms

        Example:
            >>> import numpy as np
            >>> neuron = QuantumSpikingNeuron("sim_test")
            >>> t = np.linspace(0, 100, 1000)  # 100ms simulation
            >>> inputs = 10 * np.sin(2 * np.pi * t / 50)  # Oscillatory input
            >>> results = neuron.simulate(inputs)
            >>> print(f"Spike rate: {results['spike_rate']:.2f} Hz")
        """
        dt = dt or self.config.dt
        input_currents = np.asarray(input_currents, dtype=np.float64)

        n_steps = len(input_currents)
        membrane_trace = np.zeros(n_steps)
        quantum_trace = np.zeros((n_steps, 2), dtype=np.complex128)
        spike_times = []

        logger.info(f"Starting simulation of {n_steps} steps ({n_steps * dt:.1f} ms)")

        for i, I in enumerate(input_currents):
            spike, state = self.step(I, dt)
            membrane_trace[i] = self._membrane_potential
            quantum_trace[i] = state.amplitudes

            if spike:
                spike_times.append(self._time_elapsed)

        results = {
            "spike_times": np.array(spike_times),
            "spike_count": len(spike_times),
            "spike_rate": len(spike_times) / (n_steps * dt / 1000.0),
            "membrane_trace": membrane_trace,
            "quantum_trace": quantum_trace,
            "final_state": self._quantum_state,
            "simulation_time": n_steps * dt,
            "average_computation_time": self._computation_time / n_steps,
        }

        logger.info(
            f"Simulation complete: {results['spike_count']} spikes, "
            f"{results['spike_rate']:.2f} Hz"
        )

        return results

    def reset(self) -> None:
        """Reset neuron to initial state.

        Clears all state variables but preserves configuration.
        """
        self._membrane_potential = self.config.resting_potential
        self._quantum_state = QuantumState(np.array([1.0, 0.0]))
        self._spike_history.clear()
        self._time_elapsed = 0.0
        self._is_refractory = False
        self._refractory_end_time = 0.0
        self._spike_count = 0
        self._computation_time = 0.0
        self._integration_steps = 0

        logger.info(f"Neuron {self.neuron_id} reset to initial state")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive neuron statistics.

        Returns:
            Dictionary with neuron statistics and state information
        """
        return {
            "neuron_id": self.neuron_id,
            "time_elapsed_ms": self._time_elapsed,
            "spike_count": self._spike_count,
            "spike_rate_hz": self.spike_rate,
            "current_membrane_potential_mv": self._membrane_potential,
            "current_quantum_coherence": self._quantum_state.coherence,
            "current_quantum_entropy": self._quantum_state.von_neumann_entropy(),
            "is_refractory": self._is_refractory,
            "integration_steps": self._integration_steps,
            "average_step_time_ms": self._computation_time
            / max(1, self._integration_steps)
            * 1000,
            "config": {
                "resting_potential": self.config.resting_potential,
                "threshold_potential": self.config.threshold_potential,
                "membrane_time_constant": self.config.membrane_time_constant,
                "quantum_tunneling": self.config.quantum_tunneling,
                "coherence_time": self.config.coherence_time,
            },
        }

    def __repr__(self) -> str:
        return (
            f"QuantumSpikingNeuron({self.neuron_id}, "
            f"V={self._membrane_potential:.2f}mV, "
            f"spikes={self._spike_count}, "
            f"t={self._time_elapsed:.2f}ms)"
        )


# =============================================================================
# UNIT TESTS
# =============================================================================


def test_quantum_state_normalization():
    """Test that quantum states maintain normalization."""
    state = QuantumState(np.array([1.0, 1.0]))
    assert np.isclose(state.probability_ground + state.probability_excited, 1.0)
    print("✓ Quantum state normalization test passed")


def test_quantum_state_unitary():
    """Test unitary transformations preserve norm."""
    state = QuantumState(np.array([1.0, 0.0]))

    # Rotation gate
    theta = np.pi / 4
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    new_state = state.apply_unitary(U)
    assert np.isclose(new_state.probability_ground + new_state.probability_excited, 1.0)
    print("✓ Unitary transformation test passed")


def test_neuron_integration():
    """Test basic neuron integration."""
    config = NeuronConfiguration(dt=0.1)
    neuron = QuantumSpikingNeuron("test_neuron", config)

    # Step with zero input
    spike, state = neuron.step(0.0)
    assert not spike
    assert neuron.membrane_potential < config.threshold_potential
    print("✓ Basic integration test passed")


def test_neuron_spike_generation():
    """Test that strong input generates spikes."""
    config = NeuronConfiguration(dt=0.1)
    neuron = QuantumSpikingNeuron("test_neuron", config)

    # Strong input should cause spike
    spike_generated = False
    for _ in range(100):
        spike, _ = neuron.step(20.0)  # Strong current
        if spike:
            spike_generated = True
            break

    assert spike_generated, "Neuron should spike with strong input"
    print("✓ Spike generation test passed")


def test_refractory_period():
    """Test refractory period prevents multiple spikes."""
    config = NeuronConfiguration(dt=0.1, refractory_period=5.0)
    neuron = QuantumSpikingNeuron("test_neuron", config)

    # Force a spike
    while True:
        spike, _ = neuron.step(50.0)
        if spike:
            break

    # During refractory period, shouldn't spike again immediately
    initial_count = neuron.spike_count
    for _ in range(20):  # 2 ms at dt=0.1
        neuron.step(50.0)

    # Should have exactly 1 spike (no additional during refractory)
    assert neuron.spike_count == initial_count, (
        f"Got {neuron.spike_count} spikes, expected {initial_count}"
    )
    print("✓ Refractory period test passed")


def test_quantum_coherence_decay():
    """Test that quantum coherence decays over time."""
    config = NeuronConfiguration(coherence_time=10.0, dt=0.1)
    neuron = QuantumSpikingNeuron("test_neuron", config)

    # Start with superposition
    neuron._quantum_state = QuantumState(np.array([1.0, 1.0]) / np.sqrt(2))
    initial_coherence = neuron.quantum_state.coherence

    # Run for some time
    for _ in range(100):
        neuron.step(0.0)

    # Coherence should have decayed
    final_coherence = neuron.quantum_state.coherence
    assert final_coherence < initial_coherence, (
        f"Coherence should decay: {initial_coherence:.4f} -> {final_coherence:.4f}"
    )
    print("✓ Quantum coherence decay test passed")


def test_simulation():
    """Test full simulation run."""
    config = NeuronConfiguration(dt=0.1)
    neuron = QuantumSpikingNeuron("test_neuron", config)

    # Create input trace
    t = np.linspace(0, 100, 1000)  # 100ms
    inputs = 15 * np.ones_like(t)  # Constant strong input

    results = neuron.simulate(inputs)

    assert "spike_times" in results
    assert "spike_rate" in results
    assert "membrane_trace" in results
    assert len(results["membrane_trace"]) == len(inputs)
    print("✓ Simulation test passed")


def test_reset():
    """Test neuron reset functionality."""
    config = NeuronConfiguration(dt=0.1)
    neuron = QuantumSpikingNeuron("test_neuron", config)

    # Run some steps
    for _ in range(100):
        neuron.step(10.0)

    initial_time = neuron.time_elapsed
    assert initial_time > 0

    # Reset
    neuron.reset()

    assert neuron.time_elapsed == 0.0
    assert neuron.spike_count == 0
    assert neuron.membrane_potential == config.resting_potential
    print("✓ Reset test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS - Quantum Spiking Neuron")
    print("=" * 60 + "\n")

    tests = [
        test_quantum_state_normalization,
        test_quantum_state_unitary,
        test_neuron_integration,
        test_neuron_spike_generation,
        test_refractory_period,
        test_quantum_coherence_decay,
        test_simulation,
        test_reset,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


# =============================================================================
# DEMONSTRATION
# =============================================================================


def demo_basic_operation():
    """Demonstrate basic neuron operation."""
    print("\n" + "=" * 60)
    print("DEMO: Basic Quantum Spiking Neuron Operation")
    print("=" * 60 + "\n")

    # Create neuron with default configuration
    config = NeuronConfiguration(
        resting_potential=-70.0,
        threshold_potential=-55.0,
        membrane_time_constant=20.0,
        quantum_tunneling=0.1,
        coherence_time=50.0,
    )

    neuron = QuantumSpikingNeuron("demo_neuron_1", config)
    print(f"Created neuron: {neuron}")
    print(f"Initial quantum state: {neuron.quantum_state}")
    print()

    # Simulate with varying input
    print("Simulating with oscillatory input...")
    t = np.linspace(0, 200, 2000)  # 200ms
    frequency = 0.05  # Hz
    amplitude = 20.0  # nA
    inputs = amplitude * (1 + np.sin(2 * np.pi * frequency * t))

    results = neuron.simulate(inputs)

    print(f"\nSimulation Results:")
    print(f"  Simulation time: {results['simulation_time']:.1f} ms")
    print(f"  Total spikes: {results['spike_count']}")
    print(f"  Spike rate: {results['spike_rate']:.2f} Hz")
    print(f"  First 5 spike times: {results['spike_times'][:5]}")
    print(f"  Final membrane potential: {results['membrane_trace'][-1]:.2f} mV")
    print(f"  Final quantum coherence: {results['final_state'].coherence:.4f}")
    print(
        f"  Average computation time: {results['average_computation_time'] * 1e6:.2f} μs/step"
    )

    # Get statistics
    stats = neuron.get_statistics()
    print(f"\nNeuron Statistics:")
    print(f"  Integration steps: {stats['integration_steps']}")
    print(f"  Quantum entropy: {stats['current_quantum_entropy']:.4f} bits")
    print(f"  Average step time: {stats['average_step_time_ms']:.4f} ms")


def demo_quantum_effects():
    """Demonstrate quantum mechanical effects."""
    print("\n" + "=" * 60)
    print("DEMO: Quantum Mechanical Effects")
    print("=" * 60 + "\n")

    print("Comparing neurons with different quantum parameters...\n")

    # Neuron with high coherence
    config_high = NeuronConfiguration(
        coherence_time=200.0, quantum_tunneling=0.2, dt=0.1
    )
    neuron_high = QuantumSpikingNeuron("high_coherence", config_high)

    # Neuron with low coherence
    config_low = NeuronConfiguration(
        coherence_time=10.0, quantum_tunneling=0.05, dt=0.1
    )
    neuron_low = QuantumSpikingNeuron("low_coherence", config_low)

    # Simulate both with same input
    t = np.linspace(0, 100, 1000)
    inputs = 15.0 * np.ones_like(t)

    print("Running simulations...")
    results_high = neuron_high.simulate(inputs.copy())
    results_low = neuron_low.simulate(inputs.copy())

    print(f"\nHigh Coherence Neuron (T₂={config_high.coherence_time}ms):")
    print(f"  Spikes: {results_high['spike_count']}")
    print(f"  Final coherence: {results_high['final_state'].coherence:.4f}")
    print(
        f"  Quantum entropy: {results_high['final_state'].von_neumann_entropy():.4f} bits"
    )

    print(f"\nLow Coherence Neuron (T₂={config_low.coherence_time}ms):")
    print(f"  Spikes: {results_low['spike_count']}")
    print(f"  Final coherence: {results_low['final_state'].coherence:.4f}")
    print(
        f"  Quantum entropy: {results_low['final_state'].von_neumann_entropy():.4f} bits"
    )

    print("\nObservation: Lower coherence time leads to faster decoherence")
    print("             and reduced quantum effects.")


def demo_spike_patterns():
    """Demonstrate different spike patterns."""
    print("\n" + "=" * 60)
    print("DEMO: Spike Pattern Generation")
    print("=" * 60 + "\n")

    config = NeuronConfiguration(dt=0.1)

    patterns = {
        "Constant input": lambda t: 15.0 * np.ones_like(t),
        "Oscillatory": lambda t: 20.0 * np.sin(2 * np.pi * t / 50),
        "Pulse train": lambda t: 30.0 * ((t % 40) < 10),
        "Random": lambda t: 10.0 + 10.0 * np.random.randn(len(t)),
    }

    t = np.linspace(0, 200, 2000)

    for name, input_func in patterns.items():
        neuron = QuantumSpikingNeuron(
            f"pattern_{name.lower().replace(' ', '_')}", config
        )
        inputs = input_func(t)
        results = neuron.simulate(inputs)

        print(
            f"{name:20s}: {results['spike_count']:3d} spikes "
            f"({results['spike_rate']:5.2f} Hz)"
        )


def demo_mathematical_formalism():
    """Demonstrate mathematical formalism and properties."""
    print("\n" + "=" * 60)
    print("DEMO: Mathematical Formalism")
    print("=" * 60 + "\n")

    print("Quantum State Properties:")
    print("-" * 40)

    # Create various quantum states
    states = [
        ("|0⟩ (ground)", np.array([1.0, 0.0])),
        ("|1⟩ (excited)", np.array([0.0, 1.0])),
        ("|+⟩ (superposition)", np.array([1.0, 1.0]) / np.sqrt(2)),
        ("|−⟩ (minus)", np.array([1.0, -1.0]) / np.sqrt(2)),
        ("|i⟩ (imaginary)", np.array([1.0, 1.0j]) / np.sqrt(2)),
    ]

    for name, amplitudes in states:
        state = QuantumState(amplitudes)
        rho = state.density_matrix()

        print(f"\n{name}:")
        print(f"  Amplitudes: α={state.alpha:.3f}, β={state.beta:.3f}")
        print(
            f"  Probabilities: P(0)={state.probability_ground:.3f}, "
            f"P(1)={state.probability_excited:.3f}"
        )
        print(f"  Coherence: C={state.coherence:.4f}")
        print(f"  Entropy: S={state.von_neumann_entropy():.4f} bits")
        print(f"  Density matrix:")
        print(f"    ρ = [{rho[0, 0]:.3f}  {rho[0, 1]:.3f}]")
        print(f"        [{rho[1, 0]:.3f}  {rho[1, 1]:.3f}]")

    print("\n" + "-" * 40)
    print("\nHamiltonian Evolution:")
    print("-" * 40)

    # Demonstrate unitary evolution
    state = QuantumState(np.array([1.0, 0.0]))
    print(f"\nInitial state: {state}")

    # Pauli X gate (bit flip)
    X = np.array([[0, 1], [1, 0]])
    state_x = state.apply_unitary(X)
    print(f"After Pauli-X: {state_x}")

    # Hadamard gate (superposition)
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    state_h = state.apply_unitary(H)
    print(f"After Hadamard: {state_h}")

    # Rotation around Z
    theta = np.pi / 4
    Rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]])
    state_rz = state_h.apply_unitary(Rz)
    print(f"After Rz(π/4): {state_rz}")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 70)
    print(" " * 15 + "QUANTUM SPIKING NEURON")
    print(" " * 10 + "Production Grade Implementation")
    print("=" * 70)

    # Run tests first
    if not run_all_tests():
        print("\n⚠ WARNING: Some tests failed. Proceeding with caution.\n")

    # Run demonstrations
    demo_basic_operation()
    demo_quantum_effects()
    demo_spike_patterns()
    demo_mathematical_formalism()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
