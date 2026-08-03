import { describe, expect, test } from 'vitest';

import { createEmptyDocument, createObjectAt, setVoxelOnObject } from '../components/designer/lib/document';
import { validateDocument } from '../components/designer/lib/validation';
import { SLOPE2_UP_RIGHT_HEAVY, SLOPE3_UP_RIGHT_HEAVY } from '../components/designer/constants';

describe('validation', () => {
	test('flags overlap between objects', () => {
		const created = createObjectAt(createEmptyDocument(8, 8), 1, 1, 1);
		const overlapping = createObjectAt(created.document, 1, 1, 2);
		const issues = validateDocument(overlapping.document);
		expect(issues.some((issue) => issue.code === 'overlap')).toBe(true);
	});

	test('flags mixed slope and dynamic material in the same object', () => {
		const created = createObjectAt(createEmptyDocument(8, 8), 1, 1, 7);
		const mixed = setVoxelOnObject(created.document, created.objectId, 2, 1, 1);
		const issues = validateDocument(mixed);
		expect(issues.filter((issue) => issue.code.startsWith('slope_'))).toEqual([
			expect.objectContaining({ code: 'slope_requires_static_object' }),
		]);
	});

	test('flags invalid 2x1 slope pairing', () => {
		const doc = createEmptyDocument(8, 8);
		doc.objects.push({
			id: 'obj-1',
			name: 'ground',
			origin: { x: 1, y: 1 },
			voxels: [{ x: 0, y: 0, type: SLOPE2_UP_RIGHT_HEAVY }],
			connectivity: null,
			visible: true,
			locked: false,
		});
		const issues = validateDocument(doc);
		expect(issues.some((issue) => issue.code === 'slope2_invalid_pair')).toBe(true);
	});

	test('flags invalid 3x1 slope triplet', () => {
		const doc = createEmptyDocument(8, 8);
		doc.objects.push({
			id: 'obj-1',
			name: 'ground',
			origin: { x: 1, y: 1 },
			voxels: [{ x: 0, y: 0, type: SLOPE3_UP_RIGHT_HEAVY }],
			connectivity: null,
			visible: true,
			locked: false,
		});
		const issues = validateDocument(doc);
		expect(issues.some((issue) => issue.code === 'slope3_invalid_triplet')).toBe(true);
	});

	test('flags unknown voxel types in editor documents', () => {
		const created = createObjectAt(createEmptyDocument(8, 8), 1, 1, 1);
		created.document.objects[0]!.voxels[0]!.type = 99 as never;
		const issues = validateDocument(created.document);
		expect(issues.some((issue) => issue.code === 'unknown_voxel_type')).toBe(true);
	});

	test('does not warn when optional metadata is unset', () => {
		const doc = createEmptyDocument(8, 8);
		const issues = validateDocument(doc);
		expect(issues.some((issue) => issue.code === 'spawn_out_of_bounds')).toBe(false);
		expect(issues.some((issue) => issue.code === 'target_out_of_bounds')).toBe(false);
	});

	test('warns when set metadata points are out of bounds', () => {
		const doc = createEmptyDocument(8, 8);
		doc.metadata.spawn = { x: -1, y: 2 };
		doc.metadata.target = { x: 9, y: 2 };
		const issues = validateDocument(doc);
		expect(issues.some((issue) => issue.code === 'spawn_out_of_bounds')).toBe(true);
		expect(issues.some((issue) => issue.code === 'target_out_of_bounds')).toBe(true);
	});
});
