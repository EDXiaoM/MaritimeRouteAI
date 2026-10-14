"""Search parameters set by the user in the application.

用户在界面中设置的搜索参数（每种算法各有自己的参数）。

Until version 1.2 the parameters of the lattice and of the search methods
were constants in :mod:`maritime_route.config`; changing, for example, the
number of generations of the genetic algorithm required editing the code.
Since version 1.3 every request carries a :class:`SearchParameters` object:

* the web interface builds its parameter panels from :func:`parameter_spec`
  (served by ``GET /api/parameters``), so the fields, ranges, units and
  help texts are defined in one place - here;
* the request sends the values back (``"parameters": {...}``);
  :meth:`SearchParameters.from_dict` validates them (unknown keys are
  rejected, values are clamped to their range) and fills in the defaults;
* :meth:`SearchParameters.routing_config` and
  :meth:`SearchParameters.genetic_config` turn them into the configuration
  objects that the cost map and the search modules already use;
* the values actually used are written to the result file of every search
  (:mod:`maritime_route.export.results`), so every result can be reproduced.

The defaults equal the constants of version 1.2, so a request without
parameters gives exactly the same route as before.

Groups
------
``common``
    Lattice and planner settings shared by all methods: cell budget, margin,
    diagonal moves, near-shore band, automatic refinement, coastline check
    and the time limit.
``astar`` / ``dijkstra``
    Maximum number of steps (cells taken from the priority queue) and, for
    A*, the heuristic weight (1 = optimal A*, > 1 = weighted A*).
``dynamic``
    Maximum number of value-iteration sweeps.
``genetic``
    Population, generations, early stop, selection, crossover, mutation,
    elitism, chromosome length, land penalty and random seed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

from ..config import GENETIC, ROUTING, GeneticConfig, RoutingConfig

#: Groups in the order the interface shows them; ``common`` applies to every method.
GROUPS = ("common", "astar", "dijkstra", "dynamic", "genetic")
GROUP_TITLES = {
    "common": "Lattice and run",
    "astar": "A* search",
    "dijkstra": "Dijkstra search",
    "dynamic": "Dynamic programming",
    "genetic": "Genetic algorithm",
}


def _p(default, group, label, lo=None, hi=None, step=None, unit="", help_text="", kind=None):
    """Field metadata for one parameter (used by :func:`parameter_spec`)."""
    return {"default": default, "group": group, "label": label, "min": lo, "max": hi,
            "step": step, "unit": unit, "help": help_text,
            "type": kind or ("bool" if isinstance(default, bool)
                             else "int" if isinstance(default, int) else "float")}


#: Metadata of every field of :class:`SearchParameters`, in display order.
#: 参数元数据：默认值、所属分组、标签、取值范围、单位与说明。
FIELDS: Dict[str, Dict[str, Any]] = {
    # ---- common ---------------------------------------------------------- #
    "max_grid_cells": _p(
        ROUTING.max_grid_cells, "common", "Lattice cell budget", 10_000, 4_000_000, 10_000, "cells",
        "Largest lattice that is classified. A long voyage on a fine lattice needs more cells; "
        "if the box needs more, the cell size is coarsened by 1.5 until it fits. "
        "Classification time grows linearly with the number of cells."),
    "margin_deg": _p(
        ROUTING.margin_deg, "common", "Margin around the points", 0.0, 15.0, 0.5, "°",
        "Extra space around the rectangle of the points, so that the route can go round a "
        "headland or through a strait outside the direct box."),
    "diagonal_moves": _p(
        ROUTING.diagonal_moves, "common", "Diagonal moves (8 neighbours)", help_text=
        "On: a cell is connected to its 8 neighbours; off: only to 4 (N, S, E, W)."),
    "safety_margin_cells": _p(
        ROUTING.safety_margin_cells, "common", "Near-shore band", 0, 3, 1, "cells",
        "Width of the band of water next to land whose cost is multiplied by the "
        "near-shore penalty."),
    "proximity_penalty": _p(
        ROUTING.proximity_penalty, "common", "Near-shore penalty", 1.0, 5.0, 0.1, "×",
        "Cost multiplier of a water cell in the near-shore band; 1 = no penalty."),
    "auto_refine": _p(
        True, "common", "Automatic lattice refinement", help_text=
        "When no route exists on the lattice, retry on a finer and wider lattice."),
    "max_refinements": _p(
        3, "common", "Maximum refinements", 0, 6, 1, "",
        "Number of finer/wider retries (cell × 0.55, margin + 2° each time)."),
    "coast_guard": _p(
        True, "common", "Coastline check", help_text=
        "Check the route against the independent 1 km land mask and correct it when it "
        "crosses land."),
    "use_land_mask": _p(
        True, "common", "Land mask in the correction", help_text=
        "When the route crosses land on the coastline check, make impassable the lattice "
        "cells that the independent 1 km land mask shows as mostly land (without classifying "
        "again) and search again; the fine corridor lattice uses the mask too. Prevents "
        "routes across peninsulas and islands the classifier missed."),
    "land_mask_always": _p(
        False, "common", "Land mask from the first lattice", help_text=
        "Use the land mask on every lattice from the start, not only to correct a route "
        "that crosses land. Safer on unfamiliar coasts, but narrow straits close earlier "
        "and the lattice is refined more often."),
    "land_mask_share": _p(
        0.75, "common", "Land share that blocks a cell", 0.1, 1.0, 0.05, "",
        "A cell is blocked when at least this share of its 3 x 3 sub-samples is land on "
        "the mask. Lower = stricter; 1 = only cells entirely on land."),
    "adaptive_area": _p(
        True, "common", "Adaptive search area", help_text=
        "When no route exists and the water around a point reaches the edge of the "
        "lattice, first widen the search area (the passage lies outside it); otherwise "
        "refine the lattice (a strait is narrower than a cell)."),
    "corridor_search": _p(
        True, "common", "Two-level search (corridor)", help_text=
        "When the route found on the lattice still crosses land, search again on a "
        "finer lattice only inside a corridor around it, instead of refining the whole "
        "area: the fine cells are classified only where the route can go."),
    "corridor_km": _p(
        50.0, "common", "Corridor half-width", 5.0, 500.0, 5.0, "km",
        "Distance from the coarse route within which the fine lattice is built."),
    "time_limit_s": _p(
        0.0, "common", "Time limit", 0.0, 3600.0, 5.0, "s",
        "The search stops after this time (0 = no limit). The Stop button works in any case."),
    # ---- A* -------------------------------------------------------------- #
    "astar_max_steps": _p(
        1_000_000, "astar", "Maximum number of steps", 0, 50_000_000, 1000, "steps",
        "One step = one cell taken from the priority queue and expanded. The search stops "
        "with 'step limit reached' when the goal is not reached within this number; "
        "0 = no limit. A complex course on a fine lattice needs 10^5 - 10^6 steps."),
    "astar_heuristic_weight": _p(
        ROUTING.heuristic_weight, "astar", "Heuristic weight", 1.0, 5.0, 0.1, "×",
        "1 = classical A*, the route is optimal. Above 1 = weighted A*: fewer steps, the "
        "route costs at most this factor times the optimum."),
    # ---- Dijkstra ---------------------------------------------------------- #
    "dijkstra_max_steps": _p(
        1_000_000, "dijkstra", "Maximum number of steps", 0, 50_000_000, 1000, "steps",
        "One step = one cell taken from the priority queue. Dijkstra expands every cell "
        "closer to the start than the goal, so it needs several times more steps than A*; "
        "0 = no limit."),
    # ---- dynamic programming ---------------------------------------------- #
    "dp_max_sweeps": _p(
        0, "dynamic", "Maximum number of sweeps", 0, 1_000_000, 100, "sweeps",
        "One sweep updates the Bellman equation in every cell. After k sweeps the values "
        "are exact for routes of at most k moves; 0 = until convergence."),
    # ---- genetic algorithm ------------------------------------------------ #
    "ga_population": _p(
        GENETIC.population_size, "genetic", "Population size", 10, 2000, 10, "individuals",
        "Number of routes in each generation."),
    "ga_generations": _p(
        GENETIC.generations, "genetic", "Generations (epochs)", 1, 50_000, 10, "",
        "Upper limit of generations - the step limit of the genetic algorithm."),
    "ga_stagnation": _p(
        GENETIC.stagnation_limit, "genetic", "Early stop after", 0, 50_000, 10, "generations",
        "Stop when the best route has not improved for this many generations; 0 = never "
        "stop early."),
    "ga_tournament": _p(
        GENETIC.tournament_size, "genetic", "Tournament size", 2, 20, 1, "",
        "Selection: the best of this many random individuals becomes a parent."),
    "ga_crossover": _p(
        GENETIC.crossover_rate, "genetic", "Crossover probability", 0.0, 1.0, 0.05, "",
        "Probability that a child is made from two parents (one-point crossover)."),
    "ga_mutation": _p(
        GENETIC.mutation_rate, "genetic", "Mutation probability", 0.0, 1.0, 0.05, "",
        "Probability that a child is mutated (a waypoint moved, added, removed or repaired)."),
    "ga_elite": _p(
        GENETIC.elite, "genetic", "Elite", 0, 50, 1, "individuals",
        "Best routes copied unchanged into the next generation."),
    "ga_max_waypoints": _p(
        GENETIC.max_waypoints, "genetic", "Maximum waypoints", 2, 80, 1, "",
        "Longest chromosome. A winding course (many headlands, islands) needs more "
        "waypoints."),
    "ga_land_penalty": _p(
        GENETIC.land_penalty_weight, "genetic", "Land penalty", 1.0, 100_000.0, 10.0, "",
        "Fitness weight of one land cell crossed by a route."),
    "ga_seed": _p(
        GENETIC.random_seed, "genetic", "Random seed", 0, 2_147_483_647, 1, "",
        "Same seed and parameters give the same route."),
}


@dataclass
class SearchParameters:
    """All user-settable parameters of one search. 一次搜索的全部可调参数。

    The field defaults equal the constants of :mod:`maritime_route.config`
    (version 1.2 behaviour). See :data:`FIELDS` for ranges and descriptions.
    """

    max_grid_cells: int = FIELDS["max_grid_cells"]["default"]
    margin_deg: float = FIELDS["margin_deg"]["default"]
    diagonal_moves: bool = FIELDS["diagonal_moves"]["default"]
    safety_margin_cells: int = FIELDS["safety_margin_cells"]["default"]
    proximity_penalty: float = FIELDS["proximity_penalty"]["default"]
    auto_refine: bool = True
    max_refinements: int = 3
    coast_guard: bool = True
    use_land_mask: bool = True
    land_mask_always: bool = False
    land_mask_share: float = 0.75
    adaptive_area: bool = True
    corridor_search: bool = True
    corridor_km: float = 50.0
    time_limit_s: float = 0.0
    astar_max_steps: int = FIELDS["astar_max_steps"]["default"]
    astar_heuristic_weight: float = FIELDS["astar_heuristic_weight"]["default"]
    dijkstra_max_steps: int = FIELDS["dijkstra_max_steps"]["default"]
    dp_max_sweeps: int = 0
    ga_population: int = FIELDS["ga_population"]["default"]
    ga_generations: int = FIELDS["ga_generations"]["default"]
    ga_stagnation: int = FIELDS["ga_stagnation"]["default"]
    ga_tournament: int = FIELDS["ga_tournament"]["default"]
    ga_crossover: float = FIELDS["ga_crossover"]["default"]
    ga_mutation: float = FIELDS["ga_mutation"]["default"]
    ga_elite: int = FIELDS["ga_elite"]["default"]
    ga_max_waypoints: int = FIELDS["ga_max_waypoints"]["default"]
    ga_land_penalty: float = FIELDS["ga_land_penalty"]["default"]
    ga_seed: int = FIELDS["ga_seed"]["default"]

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]]) -> "SearchParameters":
        """Validate values received from the interface. 校验并补全参数。

        Missing keys take their default; numbers are converted to the field
        type and clamped to the range of :data:`FIELDS`; booleans accept
        ``true/false``, ``1/0`` and ``"on"/"off"``.

        Raises
        ------
        ValueError
            For an unknown key or a value that is not a number.
        """
        params = cls()
        for key, raw in (values or {}).items():
            if key not in FIELDS:
                raise ValueError(f"Unknown parameter '{key}'")
            setattr(params, key, _coerce(key, raw))
        params._consistency()
        return params

    @classmethod
    def from_config(cls, config: RoutingConfig) -> "SearchParameters":
        """Defaults with the lattice values of a :class:`RoutingConfig`."""
        return cls(max_grid_cells=config.max_grid_cells, margin_deg=config.margin_deg,
                   diagonal_moves=config.diagonal_moves,
                   safety_margin_cells=config.safety_margin_cells,
                   proximity_penalty=config.proximity_penalty,
                   astar_heuristic_weight=config.heuristic_weight)

    def _consistency(self) -> None:
        """Make dependent values consistent (tournament <= population, ...)."""
        self.ga_tournament = min(self.ga_tournament, self.ga_population)
        self.ga_elite = min(self.ga_elite, self.ga_population - 1)

    def to_dict(self) -> Dict[str, Any]:
        """All values (JSON-ready)."""
        return asdict(self)

    def changed(self) -> Dict[str, Any]:
        """Only the values that differ from the defaults (for display)."""
        default = SearchParameters()
        return {k: v for k, v in asdict(self).items() if getattr(default, k) != v}

    def for_algorithm(self, algorithm: str) -> Dict[str, Any]:
        """Common values plus those of ``algorithm`` (written to result files)."""
        groups = {"common", algorithm}
        return {k: getattr(self, k) for k, meta in FIELDS.items() if meta["group"] in groups}

    def max_steps(self, algorithm: str) -> int:
        """Step limit of ``algorithm`` (0 = none): expansions, sweeps or generations."""
        return {"astar": self.astar_max_steps, "dijkstra": self.dijkstra_max_steps,
                "dynamic": self.dp_max_sweeps, "genetic": self.ga_generations}.get(algorithm, 0)

    # ------------------------------------------------------------------ #
    def routing_config(self, resolution_deg: float, margin_deg: Optional[float] = None,
                       base: RoutingConfig = ROUTING) -> RoutingConfig:
        """Lattice / A* configuration for one attempt of the planner."""
        return RoutingConfig(
            grid_resolution_deg=float(resolution_deg),
            max_grid_cells=int(self.max_grid_cells),
            margin_deg=float(self.margin_deg if margin_deg is None else margin_deg),
            diagonal_moves=bool(self.diagonal_moves),
            heuristic_weight=float(self.astar_heuristic_weight),
            safety_margin_cells=int(self.safety_margin_cells),
            proximity_penalty=float(self.proximity_penalty),
        )

    def genetic_config(self) -> GeneticConfig:
        """Genetic-algorithm configuration with the user's values."""
        low, high = GENETIC.initial_waypoints
        high = max(1, min(high, self.ga_max_waypoints))
        return GeneticConfig(
            population_size=int(self.ga_population),
            generations=int(self.ga_generations),
            stagnation_limit=int(self.ga_stagnation),
            tournament_size=int(self.ga_tournament),
            crossover_rate=float(self.ga_crossover),
            mutation_rate=float(self.ga_mutation),
            elite=int(self.ga_elite),
            max_waypoints=int(self.ga_max_waypoints),
            initial_waypoints=(min(low, high), high),
            land_penalty_weight=float(self.ga_land_penalty),
            random_seed=int(self.ga_seed),
        )


def _coerce(key: str, raw: Any):
    """Convert ``raw`` to the type of ``key`` and clamp it to its range."""
    meta = FIELDS[key]
    if meta["type"] == "bool":
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "on", "yes")
        return bool(raw)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter '{key}' must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"Parameter '{key}' must be finite")
    if meta["min"] is not None:
        value = max(value, meta["min"])
    if meta["max"] is not None:
        value = min(value, meta["max"])
    return int(round(value)) if meta["type"] == "int" else float(value)


def parameter_spec() -> Dict[str, Any]:
    """Description of all parameters for the interface (``GET /api/parameters``).

    Returns
    -------
    dict
        ``{"groups": [{"id", "title", "fields": [{key, label, type, default,
        min, max, step, unit, help}]}]}`` in display order.
    """
    groups: List[Dict[str, Any]] = []
    for gid in GROUPS:
        items = [dict(key=k, **{m: v for m, v in meta.items() if m != "group"})
                 for k, meta in FIELDS.items() if meta["group"] == gid]
        groups.append({"id": gid, "title": GROUP_TITLES[gid], "fields": items})
    return {"groups": groups}


# The dataclass and the metadata must describe the same fields.
assert [f.name for f in fields(SearchParameters)] == list(FIELDS), "FIELDS out of sync"
