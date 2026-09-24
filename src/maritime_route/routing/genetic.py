"""Genetic-algorithm route search over the navigation cost map.

This module is the third search method of the application, added so that a
population-based metaheuristic can be compared with the two exact graph
searches (A* and Dijkstra). The idea follows the classical use of genetic
algorithms for path planning on an occupancy grid - "finding a way through a
maze": a candidate route is encoded as a short list of intermediate waypoints,
a population of such routes is evolved by selection, crossover and mutation,
and the fittest route of the last generation is returned.

Encoding
--------
A chromosome is a list of lattice cells ``[w1, w2, ..., wk]`` (0 <= k <=
``max_waypoints``). The route it represents is the poly-line

    start -> w1 -> w2 -> ... -> wk -> goal

in which every segment is rasterised into an 8-connected chain of cells, so the
decoded route is a path on exactly the same graph that A* and Dijkstra search.

Fitness
-------
The fitness is the cost of the decoded path under the *same* edge cost that A*
uses::

    c(u, v) = d_gc(u, v) * (w(u) + w(v)) / 2

except that an impassable (land) cell is given the finite weight
``land_penalty_weight`` instead of infinity. A route that touches land is
therefore not rejected outright - evolution needs a gradient - but it is so
expensive that any navigable route beats it. Lower fitness is better. Because
the cost model is identical, the GA result can be compared with the optimum
found by A*: the relative difference is the *optimality gap* of the GA.

Operators
---------
* **Initialisation** - random chromosomes of 1..6 waypoints drawn from the
  navigable cells inside an ellipse around the start and the goal, sorted by
  their projection on the start-goal axis (this prevents loops).
* **Selection** - tournament selection of size 3.
* **Crossover** - one-point crossover: the head of one parent is joined to the
  tail of the other at a random cut point.
* **Mutation** - one of five moves, chosen at random: *segment repair*
  (insert a waypoint that detours around the first stretch of land a segment
  crosses), shift a waypoint by a few cells, insert a waypoint in the middle of
  a segment, delete a waypoint, or *waypoint repair* (move a waypoint that sits
  on land to the nearest navigable cell).
* **Elitism** - the two best individuals are copied unchanged into the next
  generation, so the best fitness never gets worse.
* **Termination** - after ``generations`` generations, or earlier when the best
  fitness has not improved for ``stagnation_limit`` generations.

Unlike A*, the genetic algorithm gives no guarantee of optimality and its
result depends on the random seed; its value lies in its generality (the
fitness can be any function of the whole route, e.g. one including weather or
fuel), not in speed on a static grid. The experimental comparison is given in
section 3.3 of the thesis.

Main objects
------------
:class:`_Evaluator`
    Decodes chromosomes into lattice paths and computes their fitness with a
    per-segment cache.
:func:`_projection`
    Position of a cell along the start-goal axis, used to keep waypoints in
    travel order.
:func:`genetic`
    The search itself; the genetic operators are nested functions that share
    the random generator, the candidate pool and the evaluator.

Reproducibility: all randomness comes from one ``random.Random`` instance
seeded with ``seed`` or ``GeneticConfig.random_seed`` (42), so a run with the
same inputs and seed returns the same route.

遗传算法寻路：染色体为中间航点序列，航线由相邻航点之间的栅格直线段组成；
适应度与 A* 使用同一代价模型（陆地格子给一个很大的有限惩罚权重），
通过锦标赛选择、单点交叉、四种变异（平移/插入/删除/修复）和精英保留进化种群。
遗传算法不保证最优，结果与 A* 的最优代价之差即为其"最优性差距"。
（注：当前实现另含"线段修复"变异，共五种变异操作。）
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import GENETIC, IMPASSABLE_COST, GeneticConfig
from .astar import OFFSETS_8, Cell, ProgressCallback, SearchResult, _cell_distance_table, _path_km
from .cost_map import CostMap, nearest_navigable

#: A chromosome: ordered list of intermediate waypoints (lattice cells), not
#: including the start and the goal. 染色体：中间航点列表（不含起终点）。
Chromosome = List[Cell]


class _Evaluator:
    """Decodes chromosomes into lattice paths and computes their cost.

    Segment costs are cached, because the same segment (pair of consecutive
    waypoints) occurs in many individuals and in many generations.

    解码染色体并计算代价；相同线段的代价会被缓存，避免重复计算。

    Attributes
    ----------
    cost_map : CostMap
        The shared cost map.
    impassable : numpy.ndarray of bool
        True for land cells.
    weights : numpy.ndarray of float64
        Cell weights with land replaced by ``land_weight``.
    step_km : numpy.ndarray
        Shape ``(3, 3, n_rows)``; ``step_km[dr + 1, dc + 1, row]`` is the
        length in km of the move ``(dr, dc)`` from ``row``.
    evaluations : int
        Number of :meth:`fitness` calls - the effort measure reported as
        ``nodes_expanded`` of the GA.
    """

    def __init__(self, cost_map: CostMap, land_weight: float) -> None:
        """Prepare the weight array, the step-length lookup and the cache.

        Parameters
        ----------
        cost_map : CostMap
            Cost map shared with A* and Dijkstra.
        land_weight : float
            Finite weight given to land cells in the fitness
            (``GeneticConfig.land_penalty_weight``, default 200, i.e. about
            100 times the coastal-sea weight).
        """
        self.cost_map = cost_map
        # Cell weights with land replaced by a large finite penalty.
        # 陆地格子用一个很大的有限权重代替无穷大，使适应度可比较。
        weights = cost_map.cost.astype(np.float64).copy()
        self.impassable = weights >= IMPASSABLE_COST
        weights[self.impassable] = land_weight
        self.weights = weights
        # step_km[dr + 1, dc + 1, row] = length of the move (dr, dc) from `row`
        # 步长表：从第 row 行沿 (dr, dc) 方向走一步的长度（千米）。
        # The dict from A* is copied into a dense array so that a whole
        # segment can be priced with one fancy-indexing operation.
        table = _cell_distance_table(cost_map)
        n_rows = cost_map.spec.n_rows
        self.step_km = np.zeros((3, 3, n_rows))
        for (dr, dc), lengths in table.items():
            self.step_km[dr + 1, dc + 1] = lengths
        # Cache: (a, b) -> (cost, land cells) of the directed segment a -> b.
        self._cache: Dict[Tuple[Cell, Cell], Tuple[float, int]] = {}
        self.evaluations = 0

    @staticmethod
    def rasterise(a: Cell, b: Cell) -> Tuple[np.ndarray, np.ndarray]:
        """Cells of the straight segment a -> b as an 8-connected chain.

        With ``n = max(|drow|, |dcol|)`` steps both coordinates advance by at
        most one cell per step, which is exactly one 8-connected move.
        线段栅格化：步数取行差与列差的较大者，每步行列各至多变化 1，即一次八邻域移动。

        The segment is sampled at ``n + 1`` equally spaced parameters
        ``t = 0, 1/n, ..., 1`` and each sample is rounded to the nearest cell
        (a DDA-style line). The dominant coordinate advances by exactly one
        cell per step, so there are no repeated cells and no gaps.

        Parameters
        ----------
        a, b : Cell
            End cells ``(row, col)``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(rows, cols)``, int64 arrays of length ``n + 1`` including both
            ends; a single cell when ``a == b``.
        """
        n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if n == 0:
            return np.array([a[0]]), np.array([a[1]])
        t = np.linspace(0.0, 1.0, n + 1)
        rows = np.rint(a[0] + (b[0] - a[0]) * t).astype(np.int64)
        cols = np.rint(a[1] + (b[1] - a[1]) * t).astype(np.int64)
        return rows, cols

    def segment(self, a: Cell, b: Cell) -> Tuple[float, int]:
        """Cost of the segment a -> b and the number of land cells on it.

        The segment is rasterised and priced move by move with the A* edge
        cost ``step_km * (w_i + w_{i+1}) / 2`` (land at ``land_weight``). The
        result is cached per ordered pair ``(a, b)``.
        计算线段代价与其经过的陆地格子数（结果缓存）。

        Parameters
        ----------
        a, b : Cell
            Segment end cells.

        Returns
        -------
        tuple
            ``(cost, land)`` - cost in weighted km and the number of land
            cells on the segment, both end cells included.
        """
        key = (a, b)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rows, cols = self.rasterise(a, b)
        if len(rows) < 2:
            # Zero-length segment: no cost, but the cell itself may be land.
            result = (0.0, int(self.impassable[rows[0], cols[0]]))
        else:
            w = self.weights[rows, cols]
            # Move directions along the chain, each component in {-1, 0, 1}.
            dr = np.diff(rows)
            dc = np.diff(cols)
            # Step length of every move, looked up by direction and source row.
            steps = self.step_km[dr + 1, dc + 1, rows[:-1]]
            # Sum of d * (w_i + w_{i+1}) / 2 over all moves (trapezoidal rule).
            cost = float(np.sum(steps * 0.5 * (w[:-1] + w[1:])))
            land = int(self.impassable[rows, cols].sum())
            result = (cost, land)
        self._cache[key] = result
        return result

    def fitness(self, start: Cell, goal: Cell, chromosome: Chromosome) -> Tuple[float, int]:
        """Total cost and total number of land cells of the decoded route.

        Sums :meth:`segment` over ``start -> w1 -> ... -> wk -> goal`` and
        increments :attr:`evaluations`. A waypoint on land is counted in both
        adjacent segments; the land count is only compared with zero, so the
        double count is harmless.
        适应度：整条航线的总代价与陆地格子数。

        Parameters
        ----------
        start, goal : Cell
            Fixed end cells.
        chromosome : Chromosome
            Intermediate waypoints.

        Returns
        -------
        tuple
            ``(cost, land)``; lower cost is better, ``land == 0`` means the
            route is navigable.
        """
        self.evaluations += 1
        points = [start, *chromosome, goal]
        total, land = 0.0, 0
        for a, b in zip(points[:-1], points[1:]):
            c, l = self.segment(a, b)
            total += c
            land += l
        return total, land

    def decode(self, start: Cell, goal: Cell, chromosome: Chromosome) -> List[Cell]:
        """Full lattice path of a chromosome, without repeated cells.

        Concatenates the rasterised segments. The first cell of each segment
        equals the last cell of the previous one and is skipped; a cell equal
        to its predecessor (zero-length segment) is dropped as well.
        将染色体解码为完整的网格路径（去除相邻重复格子）。

        Parameters
        ----------
        start, goal : Cell
            Fixed end cells.
        chromosome : Chromosome
            Intermediate waypoints.

        Returns
        -------
        list of Cell
            8-connected path from ``start`` to ``goal``.
        """
        points = [start, *chromosome, goal]
        path: List[Cell] = [start]
        for a, b in zip(points[:-1], points[1:]):
            rows, cols = self.rasterise(a, b)
            for r, c in zip(rows[1:].tolist(), cols[1:].tolist()):
                if (r, c) != path[-1]:
                    path.append((r, c))
        return path


def _projection(start: Cell, goal: Cell):
    """Return a function giving the position of a cell along start -> goal (0..1).

    The position is the scalar projection of ``cell - start`` on the vector
    ``goal - start`` divided by its squared length, so the start maps to 0,
    the goal to 1, and cells beside the axis to their foot point. Values
    outside [0, 1] occur for cells behind the start or beyond the goal.
    Sorting waypoints by this value keeps a route moving forward and prevents
    loops. Computed in cell units (row/col), which is sufficient for ordering.
    返回一个函数：计算格子在起点->终点方向上的投影位置（起点 0，终点 1）。

    Parameters
    ----------
    start, goal : Cell
        End cells of the voyage.

    Returns
    -------
    callable
        ``along(cell) -> float``.
    """
    ax, ay = goal[0] - start[0], goal[1] - start[1]
    # "or 1.0" guards against start == goal (zero-length axis).
    norm = float(ax * ax + ay * ay) or 1.0

    def along(cell: Cell) -> float:
        """Normalised projection of ``cell`` on the start-goal axis.

        Parameters
        ----------
        cell : Cell
            ``(row, col)``.

        Returns
        -------
        float
            0 at the start, 1 at the goal.
        """
        return ((cell[0] - start[0]) * ax + (cell[1] - start[1]) * ay) / norm

    return along


def genetic(
    cost_map: CostMap,
    start: Cell,
    goal: Cell,
    config: GeneticConfig = GENETIC,
    progress: ProgressCallback = None,
    seed: Optional[int] = None,
) -> SearchResult:
    """Search a low-cost navigable route with a genetic algorithm.

    Parameters
    ----------
    cost_map:
        Classified lattice with traversal costs (shared with A* and Dijkstra).
    start, goal:
        Navigable lattice cells (row, col).
    config:
        Population size, rates, limits - see :class:`~maritime_route.config.GeneticConfig`.
    progress:
        Optional callback ``progress(stage, fraction, details)`` used by the web
        client to show the generation counter in real time.
    seed:
        Random seed; ``None`` takes ``config.random_seed``.

    Returns
    -------
    SearchResult
        ``found`` is ``True`` only when the best route does not touch land.
        ``nodes_expanded`` holds the number of fitness evaluations, so that the
        effort of the GA can be put next to the node counts of A* and Dijkstra.

    用遗传算法搜索低代价的可航行路线。返回结果中 nodes_expanded 记录适应度评估次数。

    Notes
    -----
    ``nodes_generated`` holds the number of generations actually run and
    ``history`` the best fitness after each generation (index 0 = initial
    population), which ``scripts/make_figures.py`` plots as the convergence
    curve. ``total_cost`` is ``inf`` when the best route still touches land.
    """
    t0 = time.perf_counter()
    rng = random.Random(config.random_seed if seed is None else seed)
    spec = cost_map.spec
    n_rows, n_cols = spec.n_rows, spec.n_cols
    passable = cost_map.passable

    if not passable[start] or not passable[goal]:
        return SearchResult(found=False, algorithm="genetic",
                            runtime_s=time.perf_counter() - t0,
                            message="Start or goal cell is not navigable")

    evaluator = _Evaluator(cost_map, config.land_penalty_weight)
    along = _projection(start, goal)

    # ---- candidate cells for random waypoints ---------------------------- #
    # Navigable cells inside an ellipse whose foci are the start and the goal;
    # the ellipse is widened until it holds a reasonable number of cells.
    # 随机航点候选：以起点和终点为焦点的椭圆内的可航行格子。
    # A cell P lies inside the ellipse when |P - start| + |P - goal| <=
    # factor * |goal - start| (+ 2 cells so that very short voyages still get
    # candidates). factor 1.3 allows detours of up to 30 % of the direct
    # distance; the last factor (1e9) accepts every navigable cell.
    rr, cc = np.nonzero(passable)
    d_direct = math.hypot(goal[0] - start[0], goal[1] - start[1])
    d_sum = np.hypot(rr - start[0], cc - start[1]) + np.hypot(rr - goal[0], cc - goal[1])
    candidates = np.empty(0, dtype=np.int64)
    for factor in (1.3, 1.6, 2.0, 3.0, 1e9):
        candidates = np.nonzero(d_sum <= factor * d_direct + 2.0)[0]
        if len(candidates) >= 50:
            break
    pool: List[Cell] = [(int(rr[i]), int(cc[i])) for i in candidates]

    def random_cell() -> Cell:
        """Uniformly random navigable cell from the ellipse pool.

        Returns
        -------
        Cell
            A candidate waypoint.
        """
        return pool[rng.randrange(len(pool))]

    def ordered(cells: Sequence[Cell]) -> Chromosome:
        """Sort waypoints by their position along the start-goal axis.

        Keeps every chromosome in travel order so that operators cannot
        create routes that double back on themselves.

        Parameters
        ----------
        cells : sequence of Cell
            Waypoints in any order.

        Returns
        -------
        Chromosome
            The same cells sorted by :func:`_projection`.
        """
        return sorted(cells, key=along)

    def random_chromosome() -> Chromosome:
        """New random individual with ``initial_waypoints`` (1..6) waypoints.

        Returns
        -------
        Chromosome
            Ordered list of random pool cells.
        """
        low, high = config.initial_waypoints
        return ordered(random_cell() for _ in range(rng.randint(low, high)))

    def score(ch: Chromosome) -> float:
        """Fitness value used for selection (lower is better).

        Parameters
        ----------
        ch : Chromosome
            Individual to evaluate.

        Returns
        -------
        float
            Route cost in weighted km, land priced at ``land_penalty_weight``.
        """
        # Land is already priced in through land_penalty_weight. 陆地已通过惩罚权重计入。
        return evaluator.fitness(start, goal, ch)[0]

    # ---- genetic operators --------------------------------------------- #
    def tournament(population: List[Chromosome], fitness: List[float]) -> Chromosome:
        """Tournament selection: best of ``tournament_size`` random individuals.

        Selection pressure grows with the tournament size; size 3 keeps
        weaker individuals in play and so preserves diversity.
        锦标赛选择：随机抽取若干个体，取其中适应度最好者。

        Parameters
        ----------
        population : list of Chromosome
            Current generation.
        fitness : list of float
            Fitness of each individual, same order.

        Returns
        -------
        Chromosome
            The winner (the list object itself, not a copy).
        """
        best = min(rng.sample(range(len(population)), config.tournament_size),
                   key=lambda i: fitness[i])
        return population[best]

    def crossover(a: Chromosome, b: Chromosome) -> Chromosome:
        """One-point crossover along the start-goal axis.

        A random cut position in [0, 1) along the axis is drawn; the child
        takes the waypoints of ``a`` before the cut and those of ``b`` after
        it, so the child stays ordered. It is truncated to ``max_waypoints``.

        Parameters
        ----------
        a, b : Chromosome
            Parents.

        Returns
        -------
        Chromosome
            New child list.
        """
        # One-point crossover on the start-goal axis: head of `a`, tail of `b`.
        # 单点交叉：沿起终点方向取 a 的前半段与 b 的后半段。
        cut = rng.random()
        head = [w for w in a if along(w) <= cut]
        tail = [w for w in b if along(w) > cut]
        return (head + tail)[: config.max_waypoints]

    def clamp(cell: Cell) -> Cell:
        """Clip a cell index into the lattice bounds.

        Parameters
        ----------
        cell : Cell
            Possibly out-of-range ``(row, col)``.

        Returns
        -------
        Cell
            ``(row, col)`` with ``0 <= row < n_rows``, ``0 <= col < n_cols``.
        """
        return (min(max(cell[0], 0), n_rows - 1), min(max(cell[1], 0), n_cols - 1))

    def repair_segment(ch: Chromosome) -> Chromosome:
        """Insert a waypoint that detours around the first land run of a segment.

        The land cells of every segment are known from rasterisation. For a
        segment that crosses land, the middle cell of its first land run is
        snapped to the nearest navigable cell and inserted as a new waypoint,
        which splits the blocked segment in two. Repeated over generations this
        lets the route wrap around headlands and thread narrow straits.
        线段修复：对穿越陆地的线段，取第一段陆地的中点，吸附到最近的可航行格子并作为新航点插入。

        Parameters
        ----------
        ch : Chromosome
            Individual to repair (not modified).

        Returns
        -------
        Chromosome
            Repaired, re-ordered copy; ``ch`` itself when no segment is
            blocked, the chromosome is already at ``max_waypoints``, or no
            navigable cell is found near the land run.
        """
        points = [start, *ch, goal]
        # Indices j of blocked segments points[j] -> points[j + 1].
        blocked = [j for j in range(len(points) - 1)
                   if evaluator.segment(points[j], points[j + 1])[1] > 0]
        if not blocked or len(ch) >= config.max_waypoints:
            return ch
        j = rng.choice(blocked)
        rows, cols = evaluator.rasterise(points[j], points[j + 1])
        land = np.nonzero(evaluator.impassable[rows, cols])[0]
        # First contiguous run of land cells on the segment. 线段上第一段连续陆地。
        run_end = land[0]
        while run_end + 1 < len(rows) and evaluator.impassable[rows[run_end + 1], cols[run_end + 1]]:
            run_end += 1
        mid = (land[0] + run_end) // 2
        snapped = nearest_navigable(cost_map, int(rows[mid]), int(cols[mid]))
        if snapped is None:
            return ch
        out = list(ch)
        # Segment j runs from points[j] (= ch[j - 1]) to points[j + 1] (= ch[j]),
        # so inserting at index j places the new waypoint between them;
        # ordered() then restores the axis order.
        out.insert(j, snapped)
        return ordered(out)

    def mutate(ch: Chromosome) -> Chromosome:
        """Apply one randomly chosen mutation. 随机施加一种变异。

        One uniform draw ``move`` in [0, 1) selects the operator. Nominal
        probabilities: segment repair 25 %, shift 25 %, insert 18 %,
        delete 16 %, waypoint repair 16 %. A branch whose precondition fails
        (e.g. shift on an empty chromosome) falls through to the next
        ``elif``, so the actual shares vary slightly with the chromosome.

        Parameters
        ----------
        ch : Chromosome
            Individual to mutate (not modified).

        Returns
        -------
        Chromosome
            Mutated, re-ordered copy.
        """
        ch = list(ch)
        move = rng.random()
        if move < 0.25:
            return repair_segment(ch)
        if ch and move < 0.50:
            # Shift a waypoint by a few cells. 平移一个航点。
            # Radius: 5 % of the direct start-goal distance, at least 2 cells.
            i = rng.randrange(len(ch))
            radius = max(2, int(0.05 * d_direct))
            ch[i] = clamp((ch[i][0] + rng.randint(-radius, radius),
                           ch[i][1] + rng.randint(-radius, radius)))
        elif move < 0.68 and len(ch) < config.max_waypoints:
            # Insert a waypoint near the middle of a random segment. 插入航点。
            # The new point is scattered around the midpoint by up to half the
            # segment length, so it can pull the route sideways off the line.
            points = [start, *ch, goal]
            j = rng.randrange(len(points) - 1)
            a, b = points[j], points[j + 1]
            span = max(2, int(0.5 * math.hypot(b[0] - a[0], b[1] - a[1])))
            mid = clamp(((a[0] + b[0]) // 2 + rng.randint(-span, span),
                         (a[1] + b[1]) // 2 + rng.randint(-span, span)))
            ch.insert(j, mid)
        elif ch and move < 0.84:
            # Delete a waypoint - shortens routes once the detours are no longer needed.
            # 删除航点：绕行不再需要时缩短航线。
            del ch[rng.randrange(len(ch))]
        elif ch:
            # Repair: snap a waypoint on land to the nearest navigable cell.
            # 修复：把落在陆地上的航点吸附到最近的可航行格子。
            on_land = [i for i, w in enumerate(ch) if not passable[w]]
            if on_land:
                i = rng.choice(on_land)
                snapped = nearest_navigable(cost_map, *ch[i])
                if snapped is not None:
                    ch[i] = snapped
        return ordered(ch)

    # ---- evolution ------------------------------------------------------ #
    population: List[Chromosome] = [[]]                 # the direct line is always tried
    population += [random_chromosome() for _ in range(config.population_size - 1)]
    fitness = [score(ch) for ch in population]
    best_i = min(range(len(population)), key=lambda i: fitness[i])
    # Best individual seen so far (a copy) and its fitness. 迄今最优个体。
    best, best_fit = list(population[best_i]), fitness[best_i]
    history: List[float] = [best_fit]
    stagnant = 0            # generations since the last improvement 未改进的代数
    generation = 0

    for generation in range(1, config.generations + 1):
        # Elitism: the `elite` best individuals pass unchanged (as copies).
        # 精英保留：最优的若干个体原样进入下一代。
        order = sorted(range(len(population)), key=lambda i: fitness[i])
        next_population = [list(population[i]) for i in order[: config.elite]]
        # Fill the rest by selection -> (crossover) -> (mutation).
        while len(next_population) < config.population_size:
            parent_a = tournament(population, fitness)
            if rng.random() < config.crossover_rate:
                child = crossover(parent_a, tournament(population, fitness))
            else:
                child = list(parent_a)
            if rng.random() < config.mutation_rate:
                child = mutate(child)
            next_population.append(child)
        population = next_population
        fitness = [score(ch) for ch in population]

        gen_best = min(range(len(population)), key=lambda i: fitness[i])
        # 1e-9 tolerance: a change caused only by rounding is not an improvement.
        if fitness[gen_best] < best_fit - 1e-9:
            best, best_fit = list(population[gen_best]), fitness[gen_best]
            stagnant = 0
        else:
            stagnant += 1
        history.append(best_fit)

        if progress is not None and generation % 10 == 0:
            progress("genetic", generation / config.generations,
                     {"generation": generation, "best_cost": round(best_fit, 1)})
        # Early stop after `stagnation_limit` generations without improvement.
        # 连续若干代无改进则提前终止。
        if stagnant >= config.stagnation_limit:
            break

    # ---- result ---------------------------------------------------------- #
    cells = evaluator.decode(start, goal, best)
    # Re-evaluate only for the land count (this also adds one evaluation).
    _, land = evaluator.fitness(start, goal, best)
    coords = [spec.cell_to_coord(r, c) for r, c in cells]
    found = land == 0
    return SearchResult(
        found=found,
        path_cells=cells,
        path_coords=coords,
        total_cost=best_fit if found else float("inf"),
        distance_km=_path_km(coords),
        nodes_expanded=evaluator.evaluations,
        nodes_generated=generation,
        runtime_s=time.perf_counter() - t0,
        algorithm="genetic",
        history=history,
        message=(f"Best route after {generation} generations "
                 f"({len(best)} waypoints)" if found else
                 f"No navigable route evolved in {generation} generations"),
    )
