# User guide (Appendix A of the explanatory note)

## 1. Starting the application

Run `python scripts/run_server.py` and open <http://127.0.0.1:8000>.

The header shows two indicators:

* **live connection** — the real-time channel to the server is open. While it
  reads *reconnecting…* the application still works through ordinary HTTP
  requests, only without live progress.
* **model 96.6% acc** — the loaded classifier and its test-set accuracy.

The window is split into the control panel (left) and the chart (right). The
panel has five tabs: **Plan**, **Data**, **Compare**, **Model**, **History**.

## 2. Planning a route

### 2.1 Setting departure and destination

Three equivalent ways:

* **Click the chart.** The first click sets the departure (marker **A**), the
  second the destination (marker **B**). The hint above the coordinate fields
  says which one the next click will set.
* **Drag the markers.** Both markers can be dragged; the coordinate fields
  follow.
* **Type the coordinates** into the four fields, or press one of the
  **preset** chips (Kiel → Tallinn, Rotterdam → Lisbon, Helsinki → Copenhagen,
  Odesa → Novorossiysk).

### 2.2 Choosing the parameters

| Control | Meaning |
|---|---|
| **Search algorithm** | `A*` — heuristic search, the default; `Dijkstra` — uniform cost, same result, more nodes; `Genetic algorithm` — evolutionary search: a population of candidate routes is improved by selection, crossover and mutation under a fitness function; it gives no guarantee of optimality and is shown for comparison; `Dynamic programming` — solves the Bellman equation for the cost to the destination of every cell (value iteration) and follows it from the departure; exact, same result as A\*; `Great circle` — the naive shortest-distance reference |
| **Grid resolution** | lattice step in degrees. Finer resolution follows narrow straits more precisely but costs more time. |
| **Service speed** | used only to convert the route length into a passage time |
| **Show zone classification layer** | overlays the classified cells on the chart |
| **Live zone read-out** | classifies the point under the pointer continuously and shows the zone, confidence and depth at the bottom-left of the chart |

### 2.3 Building the route

Press **Build route**. A progress block reports each stage:

1. *Building geographic lattice*
2. *Neural network classifying grid cells* — with the number of cells done
3. *Cost map ready* — with the share of each zone
4. *A\* search in progress* — with nodes expanded and frontier size
   (for Dijkstra the same; for the genetic algorithm *Genetic algorithm:
   evolving routes* with the generation number and the best route cost so far;
   for dynamic programming *Dynamic programming: solving the Bellman equation*
   with the sweep number and the number of values improved in it)
5. *Route ready*

If no route is found — because a strait is narrower than one lattice cell, or
because the only passage lies outside the rectangle around the two ports — the
message *Refining the lattice* appears and the application retries on a finer
and wider lattice (up to three times). A departure or destination is moved onto
the nearest navigable cell only if that cell is within 60 km; a point further
inland is reported as lying on land.

The genetic algorithm may fail to find a navigable route where A\* succeeds
(it can converge on a strait that the lattice closes). In that case the lattice
is not refined, because A\* has proved that a route exists; choose A\* instead.

### 2.4 Reading the result

The chart shows:

* the **optimised route**, coloured per zone along its length;
* the **great-circle route** as a red dashed line;
* the **zone classification layer**, if enabled.

The *Route summary* panel reports the route length, the great-circle distance,
the detour ratio, the weighted traversal cost, the estimated passage time, the
planning time, the number of expanded nodes and the number of waypoints. Below
it, the *Zone profile* bar shows how much of the route lies in each zone.

## 3. Comparing with the shortest route

The **Compare** tab is filled automatically after every route. It reports the
length of both routes, the additional distance accepted by the optimised route,
how many sample points of the great circle fall on land, and a verdict: a
great-circle route that crosses land is not navigable at all, so the extra
distance is the price of a route that can actually be sailed.

Press **Compare algorithms** to run A\*, Dijkstra, the genetic algorithm, dynamic
programming and the great circle on one shared cost map. The table reports route length, cost,
expanded nodes (for the genetic algorithm: the number of routes evaluated) and
search time for each, the node reduction of A\* against Dijkstra, and how far
the genetic algorithm lies above the optimum found by A\*, and the cost
difference between dynamic programming and A\* (zero, since both are exact). The reported times
exclude the shared cost-map construction, so they compare the search methods
only.

## 4. Working with AIS data

Open the **Data** tab and drop a `.geojson`, `.csv` or `.json` file onto the
drop zone (or click it to browse).

* CSV files must contain latitude and longitude columns; the usual spellings
  (`lat`, `lon`, `lng`, `y`, `x`, …) are recognised automatically.
* GeoJSON files must be a `FeatureCollection` of points.

The application validates the file, classifies the points and stores them in
the database. The report shows how many features were read, how many were
valid, how many were dropped and how the points are distributed over the four
zones. If the file carries `class_code` labels, the agreement between the model
and those labels is reported as well.

The classified points are drawn on the chart and can be downloaded as
**Points GeoJSON** or **Points CSV**.

The *Database* block at the bottom of the tab shows how many sessions, points,
routes and waypoints are stored and how large the database file is.

## 5. Exporting the results

After a route has been built, three buttons appear under the route summary:

| Button | File | Content |
|---|---|---|
| **GeoJSON** | `route_<id>.geojson` | route line, every waypoint with its zone, and the great-circle reference line |
| **CSV** | `route_<id>.csv` | waypoint table: sequence, coordinates, zone, weight, cumulative distance, bearing |
| **PDF report** | `route_<id>.pdf` | voyage summary, zone profile, comparison with the shortest route, model information and the waypoint list |

The GeoJSON file carries styling properties (`stroke`, `marker-color`) and opens
directly in QGIS, geojson.io and similar tools.

## 6. Inspecting the model

The **Model** tab reports the architecture, the number of input features and
parameters, the size of the reference index, and the full test-set metrics:
accuracy, macro and weighted F1, Cohen's κ and macro ROC-AUC. Below them the
confusion matrix is shown cell by cell, together with the training and
validation curves of the run that produced the deployed weights.

## 7. Stored routes

The **History** tab lists the routes saved in the database, newest first, with
the algorithm, length, creation time, number of waypoints, search time and
detour ratio. Clicking an entry draws that route on the chart again.

## 8. Keyboard and mouse summary

| Action | Result |
|---|---|
| Click on the chart | set departure, then destination |
| Drag marker **A** / **B** | move departure / destination |
| Scroll on the chart | zoom |
| Move the pointer over the chart | live zone read-out (when enabled) |
| Layer control, top-left | switch between the online chart and the on-board chart |
