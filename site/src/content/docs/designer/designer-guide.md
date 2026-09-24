---
title: Designer Guide
description: Browser-based voxel editor for building and exporting EvoGym world JSON.
---

The Designer is a browser-based editor available at `/designer/`. It lets you draw multi-object voxel worlds, validate them, and export the canonical EvoGym JSON format that the simulator and training pipelines consume. No local tooling is required. The browser stores the current draft locally between visits.

---

## Layout

The interface has three regions:

- **Top bar** — document controls: New, Import, Export, Validate, Fit view, Undo/Redo.
- **Canvas** — the main drawing area.
- **Right sidebar** — stacked panels: Document, Metadata, Objects, Validation.

On narrow screens, open **Controls** to access these panels. Selecting **Place on canvas** closes the drawer so you can place the marker immediately.

The palette bar sits above the canvas and shows the two stable voxel groups plus object action buttons when an object is selected. On smaller screens, a material picker replaces the palette buttons.

## Touch controls

- **Draw** — tap a cell or drag to paint with the chosen material.
- **Erase** — tap or drag to remove voxels.
- **Move** — drag a voxel object or marker to move it; drag empty space to pan.
- **Two fingers** — pinch to zoom and drag to pan in any mode. Lift both fingers before resuming editing.
- **Zoom buttons** — use + and − in the canvas corner; use **Fit** in the top bar to see the whole grid.

Zoom in before editing small cells. Drawing and erasing affect unlocked objects only. Undo and Redo are always available in the top bar. In landscape on a phone, the site header hides and the editing controls share one row to leave more room for the canvas.

---

## Objects

Every voxel in the world belongs to a named object. Objects are listed in the Objects panel on the right sidebar.

### Creating an object

Choose **Draw** and tap or click an empty grid cell to place the first voxel. A new object is created automatically at that position. You can also click **Add** at the top of the Objects panel to create an empty object at the origin.

### Selecting an object

Choose **Move**, then tap or click a painted voxel to select its object. Alternatively, use the **Select** button on the object's card in the Objects panel.

### Moving an object

In **Move** mode, drag an unlocked object on the canvas to move it. You can also use the arrow keys when the canvas does not have keyboard focus.

### Renaming

Click the name field on the object's card in the Objects panel and type a new name. The rename is committed on blur. Object names become the JSON keys in the exported world file — they must be unique.

### Visibility

Toggle the eye icon on the object's card to hide or show the object on the canvas. Hidden objects are excluded from the rendered view but remain in the document and are included in exports.

### Locking

Toggle the lock icon to prevent accidental edits. Locked objects cannot be painted or erased on the canvas. All other operations (rename, visibility, delete) still work.

### Deleting

Click the trash icon on the object's card. A confirmation dialog appears before the object is removed. The deletion is undoable immediately after dismissing the dialog.

---

## Drawing tools

All drawing targets the currently selected object. If no object is selected, clicking on empty canvas space creates a new one.

### Place (Draw or `Space`)

In **Draw** mode, tap or click a grid cell to paint one voxel of the active type at that position. Pressing `Space` with the cursor over the canvas places a voxel at the current cursor position.

### Paint stroke (Draw + drag)

In **Draw** mode, drag across the canvas to paint a continuous stroke. Skipped cells between pointer positions are filled. On desktop, holding `Space` also enables painting in other modes.

### Erase (Erase, `X`, or right-click)

Choose **Erase** and tap or drag over voxels. On desktop, you can also press `X` with the cursor over a voxel or right-click it. This sets the cell back to empty within the owning object.

### Rectangle fill (`Shift` + drag)

Hold `Shift` and drag to define a rectangular region. A preview outline appears during the drag. Release to fill the entire rectangle with the active voxel type.

---

## Voxel palette

The palette bar presents two stable groups. The active group is highlighted. Only one type within the active group is used for all painting operations.

| # | Key | Group | Types |
|---|-----|-------|-------|
| 1 | `1` | Materials | Rigid, Soft, Fixed |
| 2 | `2` | Actuators | H-Act, V-Act, Contractile |

Every voxel type is shown as its own labelled swatch in the palette bar, grouped under
Materials and Actuators. Click a swatch to draw with that type; the active one is
highlighted, and the status bar shows it at the bottom left. The keyboard shortcuts below
are accelerators — you never need them to reach a type.

### Switching groups

- Press `Q` / `W` (or `1` / `2`) to jump directly to Materials or Actuators, respectively.
- Press `A` to cycle to the previous group, `D` to cycle to the next.
- Press `Shift+ArrowUp` / `Shift+ArrowDown` as an alternate cycle shortcut.

### Cycling types within a group

Press `S` to advance to the next type within the active group, wrapping around.

---

## Object operations

These operations require an object to be selected. The buttons appear in the palette bar when a selection exists and are also available via keyboard shortcuts.

### Rotate (`R`)

Rotates the object 90 degrees clockwise. Imported legacy objects containing multi-cell slopes (`SLOPE2_*` or `SLOPE3_*`) are not rotatable.

### Mirror (`Ctrl+M`)

Flips the object horizontally (along the X axis).

### Duplicate (`Ctrl+D`)

Creates a copy of the selected object, offset slightly from the original. The copy becomes the new selection.

### Delete (`Delete` or `Backspace`)

Deletes the selected object immediately (no confirmation when using the keyboard shortcut). Undoable with `Ctrl+Z`.

---

## Viewport

### Pan

Choose **Move** and drag empty space to pan, or drag with two fingers in any mode. You can also drag with the middle mouse button from any canvas position.

### Zoom

Pinch with two fingers or use the canvas + and − buttons. On desktop, scroll the mouse wheel to zoom around the cursor.

### Fit (top bar)

Click **Fit** in the top bar to fit the entire grid into the visible area. This is also triggered automatically after importing a file or creating a new document.

---

## Undo / Redo

| Action | Shortcut |
|--------|----------|
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Shift+Z` or `Ctrl+Y` |

The undo stack holds up to 80 states. Undo and redo buttons in the top bar are disabled when the stack is empty in the respective direction.

---

## Grid

The grid size (width and height in cells) is set per document. Open the **Document** section in the right sidebar to edit a dimension. The change is applied when you leave the field and is tracked by undo history. Each dimension is limited to 256 cells.

To create a fresh document with a specific size, click **New** in the top bar and enter the desired width and height before confirming. This replaces the current draft.

---

## Metadata

The Metadata panel holds optional **design annotations** and preview settings. Markers record your intent; they do not configure a simulation.

- **Spawn** marks a proposed starting grid cell. It does not reposition the robot.
- **Target** marks a proposed goal grid cell. It does not define a reward or a success condition.
- **Direction** draws an intended heading at the spawn marker: north, east, south, or west. It does not rotate the robot.
- **Mirror Preview** draws reflected spawn, target, and direction markers across the grid's vertical axis. It does not change the objects or exported marker coordinates.

Click **Place on canvas**, then click a grid cell to set a spawn or target marker. In **Move** mode, you can drag an existing marker to another grid cell. The X and Y fields also accept exact coordinates.

---

Designer JSON preserves these annotations in `metadata`. The Python loader
`EvoWorld.from_json` ignores them, and `EvoWorld.to_json` omits them when saving.
Object positions in the canonical JSON determine placement when compiling a
world; built-in environments define their own spawn positions and task rules.
Keep the original designer export if you want to retain your annotations.

## Import and export

### Export

Click **Export** in the top bar. Validation runs first; if there are any errors the export is blocked and the issues are highlighted in the Validation panel. On success, a file named `evogym-world.json` is downloaded.

The exported JSON structure is:

```json
{
  "grid_width": 24,
  "grid_height": 16,
  "objects": {
    "terrain": {
      "indices": [0, 1, 2],
      "types": [5, 5, 5],
      "neighbors": {
        "0": [1],
        "1": [0, 2],
        "2": [1]
      }
    }
  },
  "metadata": {
    "spawn": null,
    "target": null,
    "direction": null,
    "mirror_enabled": false
  }
}
```

- `indices` — flat grid indices (`y * grid_width + x`) for each occupied cell, sorted ascending.
- `types` — cell type integer for each index (same order as `indices`).
- `neighbors` — for each index, the stored connectivity list within the same object, sorted ascending.
- Object keys are the names assigned in the Objects panel.
- `metadata` — optional spawn, target, and direction values plus the mirror-preview setting.

### Import

Click **Import** and select a `.json` file. The file is parsed and validated. If parsing succeeds, the current draft is replaced and the view is fitted. If parsing fails, an error message appears and the current draft is unchanged.

Legacy slope fixtures can still be imported for inspection. Import validates multi-cell slopes:

- `SLOPE2_*`: every heavy voxel must have its corresponding light mate in the correct adjacent cell (and vice versa)
- `SLOPE3_*`: every heavy/mid/light segment must have the two expected neighboring segments in the correct adjacent cells

The designer preserves imported custom connectivity during import and export. If you paint or erase a voxel in an object, the designer regenerates that object's connectivity from four-neighbor grid adjacency. Moving, rotating, mirroring, or duplicating an unedited object preserves its custom edges.

### Validate

Validation runs automatically about 600 ms after each edit. On larger screens, click **Validate** to run it immediately without exporting. On phones, open **Controls** to read the automatic validation results. Issues appear in the Validation panel with a severity, an actionable message, and the affected object when applicable.

---

## Unsupported slopes

The public designer palette exposes only stable square-cell types: `RIGID`,
`SOFT`, `FIXED`, `H_ACT`, `V_ACT`, and `CONTRACTILE`.

Slope constants and validation helpers remain in the codebase for legacy JSON
inspection, but slope painting is not part of the supported designer workflow.
Slopes are omitted from the palette because their friction/support model is
still experimental; stable worlds should use square-cell terrain.

---

## Keyboard shortcuts

Use `Cmd` in place of `Ctrl` on macOS. Canvas arrow keys move the keyboard cursor when the canvas has focus; otherwise they move the selected object.

| Action | Shortcut |
|--------|----------|
| Select group | `Q` / `W` (also `1` / `2`) |
| Cycle groups | `A` / `D` (also `Shift+ArrowUp` / `Shift+ArrowDown`) |
| Cycle type in group | `S` |
| Move selected object | `Arrow keys` |
| Place voxel | `Space` / Click |
| Paint stroke | Draw + drag (or `Space` + drag) |
| Erase voxel | `X` / Right-click |
| Rectangle fill | `Shift` + Drag |
| Pan viewport | Move + drag empty space / Middle-drag / Two-finger drag |
| Zoom | Scroll / Pinch / Zoom buttons |
| Rotate object | `R` |
| Mirror object | `Ctrl+M` |
| Duplicate object | `Ctrl+D` |
| Delete object | `Delete` / `Backspace` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Shift+Z` or `Ctrl+Y` |
| Show shortcuts | `?` |

## Run your design

Continue to [Designer to Simulation](../../guides/designer-world/) to load the exported world in Python and run a controller. The designer edits geometry and annotations; simulation and task evaluation run in Python.
