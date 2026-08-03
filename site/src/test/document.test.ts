import { describe, expect, test } from 'vitest';

import {
	createEmptyDocument,
	createObjectAt,
	deleteObject,
	duplicateObject,
	mirrorObject,
	paintRect,
	removeVoxelFromObject,
	rotateObject,
	setGridSize,
	setVoxelOnObject,
} from '../components/designer/lib/document';
import {
	RIGID,
	SOFT,
	SLOPE_UP_LEFT,
	SLOPE_UP_RIGHT,
	SLOPE2_UP_RIGHT_HEAVY,
	SLOPE2_UP_RIGHT_LIGHT,
	SLOPE2_DOWN_LEFT_HEAVY,
	SLOPE3_UP_RIGHT_HEAVY,
	SLOPE3_UP_RIGHT_LIGHT,
	SLOPE3_UP_RIGHT_MID,
	SLOPE3_DOWN_LEFT_HEAVY,
} from '../components/designer/constants';
import { objectBounds } from '../components/designer/lib/geometry';

describe('document helpers', () => {
	test('createObjectAt creates a named object with one voxel', () => {
		const result = createObjectAt(createEmptyDocument(12, 8), 3, 2, 1);
		expect(result.document.objects).toHaveLength(1);
		expect(result.document.objects[0]?.name).toBe('object_1');
		expect(result.document.objects[0]?.origin).toEqual({ x: 3, y: 2 });
	});

	test('empty documents include default metadata', () => {
		const doc = createEmptyDocument(12, 8);
		expect(doc.metadata).toEqual({
			spawn: null,
			target: null,
			direction: null,
			mirrorEnabled: false,
		});
	});

	test('grid resizing clamps dimensions at the rendering safety limit', () => {
		const resized = setGridSize(createEmptyDocument(12, 8), 99_999, 0);
		expect(resized.gridWidth).toBe(256);
		expect(resized.gridHeight).toBe(1);
	});

	test('paintRect expands the selected object across a rectangle', () => {
		const created = createObjectAt(createEmptyDocument(12, 8), 1, 1, 1);
		const painted = paintRect(created.document, created.objectId, { x: 1, y: 1 }, { x: 3, y: 2 }, 2);
		expect(painted.objects[0]?.voxels).toHaveLength(6);
		expect(objectBounds(painted.objects[0]!)).toEqual({
			minX: 1,
			minY: 1,
			maxX: 3,
			maxY: 2,
		});
	});

	test('rotate turns asymmetric geometry clockwise and remaps slope types in the same direction', () => {
		const created = createObjectAt(createEmptyDocument(), 3, 2, RIGID);
		const withArm = setVoxelOnObject(created.document, created.objectId, 4, 2, SOFT);
		const asymmetric = setVoxelOnObject(withArm, created.objectId, 3, 3, SLOPE_UP_RIGHT);
		const rotated = rotateObject(asymmetric, created.objectId);
		expect(
			rotated.objects[0]?.voxels
				.map((voxel) => [voxel.x, voxel.y, voxel.type])
				.sort((left, right) => left[0]! - right[0]! || left[1]! - right[1]!),
		).toEqual([
			[0, 0, SOFT],
			[0, 1, RIGID],
			[1, 1, SLOPE_UP_LEFT],
		]);
	});

	test('normalization prevents mirror from moving a shape after its leftmost cell is erased', () => {
		const created = createObjectAt(createEmptyDocument(), 5, 5, RIGID);
		const extended = setVoxelOnObject(created.document, created.objectId, 6, 5, RIGID);
		const erased = removeVoxelFromObject(extended, created.objectId, 5, 5);
		expect(erased.objects[0]?.origin).toEqual({ x: 6, y: 5 });
		expect(erased.objects[0]?.voxels).toEqual([{ x: 0, y: 0, type: RIGID }]);

		const mirrored = mirrorObject(erased, created.objectId);
		expect(objectBounds(mirrored.objects[0]!)).toEqual({
			minX: 6,
			minY: 5,
			maxX: 6,
			maxY: 5,
		});
	});

	test('duplicateObject offsets origin and creates a unique copy name', () => {
		const created = createObjectAt(createEmptyDocument(), 2, 3, 1);
		const duplicated = duplicateObject(created.document, created.objectId);
		expect(duplicated.document.objects).toHaveLength(2);
		expect(duplicated.document.objects[1]?.name).toBe('object_1_copy');
		expect(duplicated.document.objects[1]?.origin).toEqual({ x: 3, y: 4 });
	});

	test('deleteObject removes the selected object', () => {
		const created = createObjectAt(createEmptyDocument(), 2, 3, 1);
		const removed = deleteObject(created.document, created.objectId);
		expect(removed.objects).toHaveLength(0);
	});

	test('createObjectAt with a 2x1 heavy slope creates both paired halves', () => {
		const created = createObjectAt(createEmptyDocument(), 2, 2, SLOPE2_UP_RIGHT_HEAVY);
		const voxels = created.document.objects[0]?.voxels ?? [];
		expect(voxels).toHaveLength(2);
		const heavy = voxels.find((voxel) => voxel.type === SLOPE2_UP_RIGHT_HEAVY);
		const light = voxels.find((voxel) => voxel.type === SLOPE2_UP_RIGHT_LIGHT);
		expect(heavy).toBeTruthy();
		expect(light).toBeTruthy();
		expect((heavy?.x ?? 0) - (light?.x ?? 0)).toBe(1);
		expect(heavy?.y).toBe(light?.y);
	});

	test('erasing one half of a 2x1 slope erases the whole pair', () => {
		const created = createObjectAt(createEmptyDocument(), 5, 5, SLOPE2_UP_RIGHT_HEAVY);
		const erased = removeVoxelFromObject(created.document, created.objectId, 5, 5);
		expect(erased.objects).toHaveLength(0);
	});

	test('paintRect is disabled for 2x1 slope types', () => {
		const created = createObjectAt(createEmptyDocument(12, 8), 1, 1, 1);
		const painted = paintRect(
			created.document,
			created.objectId,
			{ x: 1, y: 1 },
			{ x: 3, y: 2 },
			SLOPE2_UP_RIGHT_HEAVY,
		);
		expect(painted).toBe(created.document);
	});

	test('rotateObject is blocked for objects that contain 2x1 slopes', () => {
		const created = createObjectAt(createEmptyDocument(), 0, 0, SLOPE2_DOWN_LEFT_HEAVY);
		const rotated = rotateObject(created.document, created.objectId);
		expect(rotated).toBe(created.document);
	});

	test('createObjectAt with a 3x1 heavy slope creates all three segments', () => {
		const created = createObjectAt(createEmptyDocument(), 4, 4, SLOPE3_UP_RIGHT_HEAVY);
		const voxels = created.document.objects[0]?.voxels ?? [];
		expect(voxels).toHaveLength(3);
		const heavy = voxels.find((voxel) => voxel.type === SLOPE3_UP_RIGHT_HEAVY);
		const mid = voxels.find((voxel) => voxel.type === SLOPE3_UP_RIGHT_MID);
		const light = voxels.find((voxel) => voxel.type === SLOPE3_UP_RIGHT_LIGHT);
		expect(heavy).toBeTruthy();
		expect(mid).toBeTruthy();
		expect(light).toBeTruthy();
		expect((heavy?.x ?? 0) - (mid?.x ?? 0)).toBe(1);
		expect((heavy?.x ?? 0) - (light?.x ?? 0)).toBe(2);
		expect(heavy?.y).toBe(mid?.y);
		expect(mid?.y).toBe(light?.y);
	});

	test('erasing one segment of a 3x1 slope erases the whole triplet', () => {
		const created = createObjectAt(createEmptyDocument(), 5, 5, SLOPE3_UP_RIGHT_HEAVY);
		const erased = removeVoxelFromObject(created.document, created.objectId, 5, 5);
		expect(erased.objects).toHaveLength(0);
	});

	test('paintRect is disabled for 3x1 slope types', () => {
		const created = createObjectAt(createEmptyDocument(12, 8), 1, 1, 1);
		const painted = paintRect(
			created.document,
			created.objectId,
			{ x: 1, y: 1 },
			{ x: 3, y: 2 },
			SLOPE3_UP_RIGHT_HEAVY,
		);
		expect(painted).toBe(created.document);
	});

	test('rotateObject is blocked for objects that contain 3x1 slopes', () => {
		const created = createObjectAt(createEmptyDocument(), 0, 0, SLOPE3_DOWN_LEFT_HEAVY);
		const rotated = rotateObject(created.document, created.objectId);
		expect(rotated).toBe(created.document);
	});
});
