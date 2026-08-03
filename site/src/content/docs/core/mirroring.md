---
title: Mirroring
description: Compile-time world mirroring and the robot override rules that matter for paired evaluation.
---

Mirroring is a **compile-time template capability**.

## API

Use:

```python
template_set = compile_world_templates(
    world,
    robot_name="robot",
    mirror_mode="paired",
)
```

This returns:

- `template_set.primary`
- `template_set.mirror`

## What gets mirrored

For each object:

- the local grid is flipped horizontally
- local voxel connectivity is mirrored
- the world-space origin is mirrored in x using the world width

The mirror is therefore a true mirrored world/template, not a post-hoc flipped render or a composited-grid trick.

## Within-genome usage

Paired mirrored templates are intended to be used **within the same genome** when you want matched primary/mirror evaluation.

Typical pattern:

1. compile a paired template set once
2. instantiate the same genome/body/controller against `primary`
3. instantiate the same genome/body/controller against `mirror`
4. aggregate or compare the results

## Important `robot_override` rule

`instantiate_world` does **not** auto-mirror `robot_override`.

That means:

- the caller chooses which override to supply to each template
- if you want the same fixed-frame override on both sides, you pass it explicitly to both
- if you want an explicitly mirrored override, you construct that override yourself

## Unsupported slopes

The old slope transform tables are still present internally for legacy fixtures,
but slope cells are unsupported in the stable public build path and are not
exposed by the public designer palette. Experimental slope compilation, when
explicitly enabled, is geometry-only and does not provide slope friction/support.
