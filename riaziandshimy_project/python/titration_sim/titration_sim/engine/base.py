"""Core data structures and the abstract engine interface.

Architecture rule (second answer, section 0): the engine is *stateless*
with respect to the UI.  The UI only talks to a :class:`BaseEngine` through
the pure function ``state_after = engine.titrate(model, state_before, dV)``.
Every titration family (acid-base now; redox, precipitation, EDTA later,
see ROADMAP.md) is one :class:`TitrationModel` subclass plus one engine.

Reference: [MATH.md section 0]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Species:
    """A chemical species taking part in a titration.

    Example
    -------
    >>> Species("HCl", 0).name
    'HCl'
    """

    name: str
    charge: int = 0


@dataclass(frozen=True)
class Reaction:
    """A reaction with stoichiometric coefficients and a family label.

    Attributes
    ----------
    reactants, products : tuple of (Species, coefficient)
    k : float or None
        Equilibrium constant (None for a quantitative reaction).
    family : str
        'acid', 'redox', 'precip' or 'complex'.

    Example
    -------
    >>> r = Reaction((), (), None, "acid")
    >>> r.family
    'acid'
    """

    reactants: tuple[tuple[Species, float], ...]
    products: tuple[tuple[Species, float], ...]
    k: float | None
    family: str


@dataclass(frozen=True)
class TitrationState:
    """Snapshot of the vessel after some titrant has been added.

    Attributes
    ----------
    v_titrant_ml : float
        Volume of titrant added so far (mL).
    volume_ml : float
        Total volume of solution (mL).
    ph : float
        pH (-log10 [H+], concentration scale).
    h : float
        Free proton concentration (mol/L).
    species : dict[str, float]
        Equilibrium concentrations (mol/L), keyed by name.
    temperature_c : float
        Temperature (deg C).

    Example
    -------
    >>> TitrationState(0.0, 25.0, 7.0, 1e-7, {}, 25.0).volume_ml
    25.0
    """

    v_titrant_ml: float
    volume_ml: float
    ph: float
    h: float
    species: dict[str, float] = field(default_factory=dict)
    temperature_c: float = 25.0


@dataclass(frozen=True)
class TitrationModel:
    """Marker base class for the chemistry description of a titration.

    Concrete subclasses add their own fields (concentrations, constants).
    """


@dataclass(frozen=True)
class Measurement:
    """Instrument layer output, separate from the chemistry (section 0).

    Example
    -------
    >>> Measurement(v_titrant_ml=1.0, ph=4.0).ph
    4.0
    """

    v_titrant_ml: float
    ph: float | None = None
    emf_mv: float | None = None
    conductivity: float | None = None
    absorbance: float | None = None


class BaseEngine(ABC):
    """Abstract titration engine.

    Subclasses must be *pure*: no hidden state, same input -> same output.
    """

    @abstractmethod
    def initial_state(self, model: TitrationModel) -> TitrationState:
        """State before any titrant is added.

        Parameters
        ----------
        model : TitrationModel
            Chemistry description.

        Returns
        -------
        TitrationState
        """

    @abstractmethod
    def titrate(
        self, model: TitrationModel, state: TitrationState, d_v_ml: float
    ) -> TitrationState:
        """Add ``d_v_ml`` mL titrant to ``state`` and return the new state.

        Raises
        ------
        ValueError
            If ``d_v_ml`` is negative.
        """

    def measure(self, state: TitrationState) -> Measurement:
        """Default measurement layer: ideal pH electrode.

        Example
        -------
        >>> from titration_sim.engine.base import TitrationState
        >>> class E(BaseEngine):
        ...     def initial_state(self, model): ...
        ...     def titrate(self, model, state, d_v_ml): ...
        >>> E().measure(TitrationState(0.0, 25.0, 3.0, 1e-3)).ph
        3.0
        """
        return Measurement(v_titrant_ml=state.v_titrant_ml, ph=state.ph)
