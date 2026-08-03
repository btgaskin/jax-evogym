import { describe, expect, test } from 'vitest';

import {
	SLOPE2_UP_RIGHT_HEAVY,
	SLOPE2_UP_RIGHT_LIGHT,
	SLOPE3_UP_RIGHT_HEAVY,
	SLOPE3_UP_RIGHT_MID,
	SLOPE3_UP_RIGHT_LIGHT,
} from '../components/designer/constants';
import { createInitialEditorState, editorReducer } from '../components/designer/state/reducer';

describe('editor reducer', () => {
	test('reads drafts from the supplied storage key', () => {
		let requestedKey: string | null = null;
		const getItem = (key: string) => {
			requestedKey = key;
			return (
			key === 'custom-designer-draft'
				? JSON.stringify({
						gridWidth: 7,
						gridHeight: 5,
						objects: [],
						metadata: {
							spawn: null,
							target: null,
							direction: null,
							mirrorEnabled: false,
						},
					})
				: null
			);
		};
		const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
		Object.defineProperty(globalThis, 'localStorage', {
			configurable: true,
			value: { getItem },
		});
		try {
			const state = createInitialEditorState('custom-designer-draft');
			expect(requestedKey).toBe('custom-designer-draft');
			expect([state.document.gridWidth, state.document.gridHeight]).toEqual([7, 5]);
		} finally {
			if (originalStorage) {
				Object.defineProperty(globalThis, 'localStorage', originalStorage);
			} else {
				Reflect.deleteProperty(globalThis, 'localStorage');
			}
		}
	});

	test('undo and redo restore document mutations', () => {
		const initial = createInitialEditorState();
		const painted = editorReducer(
			initial,
			{ type: 'PAINT_AT', worldX: 1, worldY: 1 },
		);
		expect(painted.document.objects).toHaveLength(1);
		const undone = editorReducer(painted, { type: 'UNDO' });
		expect(undone.document.objects).toHaveLength(0);
		const redone = editorReducer(undone, { type: 'REDO' });
		expect(redone.document.objects).toHaveLength(1);
	});

	test('deleting the selected object clears selection to the next valid object', () => {
		const initial = createInitialEditorState();
		const first = editorReducer(initial, { type: 'CREATE_OBJECT_AT', worldX: 0, worldY: 0 });
		const second = editorReducer(first, { type: 'CREATE_OBJECT_AT', worldX: 4, worldY: 0 });
		const selected = editorReducer(second, {
			type: 'SELECT_OBJECT',
			objectId: second.document.objects[0]!.id,
		});
		const deleted = editorReducer(selected, {
			type: 'DELETE_OBJECT',
			objectId: selected.document.objects[0]!.id,
		});
		expect(deleted.selectedObjectId).toBe(deleted.document.objects[0]!.id);
	});

	test('painting cannot add a voxel to the selected object through another object', () => {
		const initial = createInitialEditorState();
		const first = editorReducer(initial, { type: 'CREATE_OBJECT_AT', worldX: 1, worldY: 1 });
		const second = editorReducer(first, { type: 'CREATE_OBJECT_AT', worldX: 3, worldY: 1 });
		const selected = editorReducer(second, {
			type: 'SELECT_OBJECT',
			objectId: first.document.objects[0]!.id,
		});
		const blocked = editorReducer(selected, { type: 'PAINT_AT', worldX: 3, worldY: 1 });
		expect(blocked).toBe(selected);
		expect(blocked.document.objects[0]!.voxels).toHaveLength(1);
	});

	test('metadata updates participate in undo and redo history', () => {
		const initial = createInitialEditorState();
		const withSpawn = editorReducer(initial, { type: 'SET_SPAWN_POINT', point: { x: 2, y: 3 } });
		const withMirror = editorReducer(withSpawn, { type: 'SET_MIRROR_ENABLED', enabled: true });
		expect(withMirror.document.metadata.spawn).toEqual({ x: 2, y: 3 });
		expect(withMirror.document.metadata.mirrorEnabled).toBe(true);

		const undone = editorReducer(withMirror, { type: 'UNDO' });
		expect(undone.document.metadata.mirrorEnabled).toBe(false);
		expect(undone.document.metadata.spawn).toEqual({ x: 2, y: 3 });

		const redone = editorReducer(undone, { type: 'REDO' });
		expect(redone.document.metadata.mirrorEnabled).toBe(true);
	});

	test('rotate on a slope2 object emits a warning and does not mutate document', () => {
		const state = createInitialEditorState();
		const imported = editorReducer(state, {
			type: 'IMPORT_DOCUMENT',
			document: {
				gridWidth: 8,
				gridHeight: 8,
				metadata: {
					spawn: null,
					target: null,
					direction: null,
					mirrorEnabled: false,
				},
				objects: [
					{
						id: 'obj-1',
						name: 'ground',
						origin: { x: 2, y: 2 },
						voxels: [
							{ x: 0, y: 0, type: SLOPE2_UP_RIGHT_HEAVY },
							{ x: -1, y: 0, type: SLOPE2_UP_RIGHT_LIGHT },
						],
						connectivity: null,
						visible: true,
						locked: false,
					},
				],
			},
		});
		const rotated = editorReducer(imported, { type: 'ROTATE_OBJECT', objectId: 'obj-1' });
		expect(rotated.document).toBe(imported.document);
		expect(rotated.issues.some((issue) => issue.code === 'slope2_rotate_disabled')).toBe(true);
	});

	test('rotate on a slope3 object emits a warning and does not mutate document', () => {
		const state = createInitialEditorState();
		const imported = editorReducer(state, {
			type: 'IMPORT_DOCUMENT',
			document: {
				gridWidth: 8,
				gridHeight: 8,
				metadata: {
					spawn: null,
					target: null,
					direction: null,
					mirrorEnabled: false,
				},
				objects: [
					{
						id: 'obj-1',
						name: 'ground',
						origin: { x: 2, y: 2 },
						voxels: [
							{ x: 0, y: 0, type: SLOPE3_UP_RIGHT_HEAVY },
							{ x: -1, y: 0, type: SLOPE3_UP_RIGHT_MID },
							{ x: -2, y: 0, type: SLOPE3_UP_RIGHT_LIGHT },
						],
						connectivity: null,
						visible: true,
						locked: false,
					},
				],
			},
		});
		const rotated = editorReducer(imported, { type: 'ROTATE_OBJECT', objectId: 'obj-1' });
		expect(rotated.document).toBe(imported.document);
		expect(rotated.issues.some((issue) => issue.code === 'slope2_rotate_disabled')).toBe(true);
	});
});
