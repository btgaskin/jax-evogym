---
title: World JSON Format
description: The canonical EvoGym JSON format for world definitions — schema, coordinate system, and worked example.
---

Every world in jax-evogym is serialisable to and from a JSON file. The core
format is shared by the Python API, the web designer, and the original C++
EvoGym. `EvoWorld.to_json` and `EvoWorld.from_json` are the canonical Python
read/write paths.

## Top-level schema

```json
{
    "grid_width":  <int>,
    "grid_height": <int>,
    "objects": {
        "<name>": { ... },
        "<name>": { ... }
    },
    "metadata": { ... }
}
```

`metadata` is optional; the core Python serializer omits it.

| Field | Type | Description |
|-------|------|-------------|
| `grid_width` | integer | Width of the world grid in voxels. |
| `grid_height` | integer | Height of the world grid in voxels. |
| `objects` | object | Named world objects. Keys are object names; values are per-object records. |
| `metadata` | object, optional | Designer-only spawn, target, direction, and mirror annotations. They do not configure simulation behaviour. |

The declared width and height are part of the coordinate system. The Python
loader retains them even when objects occupy only a small part of the grid, so
loading and saving a sparse world does not renumber its flat indices.

## Per-object structure

Each object in `objects` contains two parallel arrays and one adjacency map.

```json
{
    "indices":   [<int>, ...],
    "types":     [<int>, ...],
    "neighbors": {
        "<index>": [<int>, ...],
        ...
    }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `indices` | `int[]` | Flat grid indices of every non-empty voxel, sorted ascending. |
| `types` | `int[]` | Cell type for each voxel, in the same order as `indices`. |
| `neighbors` | `{string: int[]}` | Stored connectivity map: key is a flat index (as a string), value is the list of connected flat indices in the same object, sorted ascending. |

`indices` and `types` are always the same length. Every index that appears in `indices` must also appear as a key in `neighbors` (even if its neighbour list is empty, e.g. an isolated voxel).

Connectivity can be hand-authored and does not have to equal four-neighbour
geometric adjacency. `EvoWorld` preserves the stored graph when it loads and
saves a world. References are object-local: a neighbour entry cannot connect
voxels belonging to different objects.

## Coordinate system

The grid uses a **y-up** convention. Row 0 is the bottom of the world; row `grid_height - 1` is the top.

A voxel at grid position `(x, y)` — where `x` counts from the left and `y` counts from the bottom — maps to the flat index:

```
flat_index = y * grid_width + x
```

And the inverse:

```
x = flat_index % grid_width
y = flat_index // grid_width
```

**Important:** `EvoWorld.add_from_array` accepts arrays in **user convention** (row 0 = top, matching how you write NumPy arrays visually). The JSON stores voxels in **internal convention** (row 0 = bottom). The conversion (`numpy.flipud`) happens automatically on load and save — you never need to flip manually when working through the Python API.

## Cell type values

See [Voxel Types](../../reference/voxel-types/) for the complete reference. The values encoded in `types` are:

| Value | Constant | Notes |
|-------|----------|-------|
| 0 | `EMPTY` | Not written to JSON — only non-empty cells appear in `indices`. |
| 1 | `RIGID` | Dynamic structural material. |
| 2 | `SOFT` | Dynamic deformable material. |
| 3 | `H_ACT` | Horizontal actuator. |
| 4 | `V_ACT` | Vertical actuator. |
| 5 | `FIXED` | Static, anchored (ground, walls, platforms). |
| 6 | `CONTRACTILE` | Universal per-axis actuator. |
| 7–30 | `SLOPE_*`, `SLOPE2_*`, `SLOPE3_*` | Legacy unsupported slope terrain values. Loadable for inspection; rejected by stable compilation unless `allow_experimental_slopes=True`. |

`allow_experimental_slopes=True` enables geometry-only inspection of legacy
slope cells. It does not enable slope friction, slope grip, or supported
shallow-ramp physics.

## Worked example

A minimal two-object world: a 4-cell fixed ground strip and a 2×1 robot with one horizontal actuator and one vertical actuator, placed on top at positions (1, 1) and (2, 1).

Grid layout (`W=4`, `H=2`, y-up):

```
row 1:  .  H  V  .       (y=1: robot at x=1,2)
row 0:  F  F  F  F       (y=0: ground at x=0,1,2,3)
         0  1  2  3
```

Flat indices:
- Ground voxels: y=0 → indices 0, 1, 2, 3
- Robot `H_ACT` at (x=1, y=1): index = 1×4 + 1 = 5
- Robot `V_ACT` at (x=2, y=1): index = 1×4 + 2 = 6

```json
{
    "grid_width": 4,
    "grid_height": 2,
    "objects": {
        "ground": {
            "indices": [0, 1, 2, 3],
            "types":   [5, 5, 5, 5],
            "neighbors": {
                "0": [1],
                "1": [0, 2],
                "2": [1, 3],
                "3": [2]
            }
        },
        "robot": {
            "indices": [5, 6],
            "types":   [3, 4],
            "neighbors": {
                "5": [6],
                "6": [5]
            }
        }
    }
}
```

The `neighbors` map stores this example's object-local voxel connectivity. The
ground and robot have separate graphs; no connection crosses the object
boundary.

## Generating and consuming JSON

### Python API

```python
import numpy as np
from jax_evogym import EvoWorld, FIXED, H_ACT, V_ACT

world = EvoWorld()
world.add_from_array("ground", np.array([[FIXED, FIXED, FIXED, FIXED]]), x=0, y=0)
world.add_from_array("robot", np.array([[H_ACT, V_ACT]]), x=1, y=1)

# Write
world.to_json("my_world.json")

# Read
loaded = EvoWorld.from_json("my_world.json")

# Get the dict without writing to disk
d = world.to_json_dict()
```

### Designer import / export

The web designer reads and writes this format. Click **Import** in the top bar to
load a JSON file, and click **Export** to download the current canvas as JSON.
The coordinate system conversion is handled automatically: the designer
displays y-up (row 0 at the bottom), matching the JSON convention.

Designer exports include an optional top-level `metadata` object with `spawn`,
`target`, `direction`, and `mirror_enabled` fields. It is a designer-only
extension: `EvoWorld.from_json` ignores it and `EvoWorld.to_json` omits it.
Keep the designer export to preserve annotations. Spawn does not reposition a
robot, target does not define a task reward, and direction does not rotate a body. Object
positions, voxel types, declared grid dimensions, and imported custom
connectivity remain in the canonical fields described above.

## Validation rules

The Python loader (`add_from_json`) enforces:

1. `grid_width` and `grid_height` must be positive integers.
2. `indices` and `types` must have equal length.
3. Indices must be unique and fall inside the declared grid.
4. Every voxel index in `indices` must also be a key in `neighbors`.
5. All type values must be known constants (0–30). Unknown values raise `ValueError`. Values 7–30 are legacy slope ids and are not stable simulation ids.
6. No two objects may share a grid cell. Overlapping objects raise `ValueError`.
7. Neighbor references must point to flat indices that appear in the same object's `indices` list.
