import { describe, expect, test } from 'vitest';

import { setVoxelOnObject } from '../components/designer/lib/document';
import { documentFromWorldJson, worldJsonFromDocument } from '../components/designer/lib/io';
import { readWorldFixture } from './fixtures';

const multiObjectFixture = {
	grid_width: 8,
	grid_height: 6,
	metadata: {
		spawn: { x: 1, y: 1 },
		target: { x: 6, y: 4 },
		direction: 'east',
		mirror_enabled: true,
	},
	objects: {
		ground: {
			indices: [0],
			types: [5],
			neighbors: {
				'0': [],
			},
		},
		robot: {
			indices: [11, 12],
			types: [1, 3],
			neighbors: {
				'11': [12],
				'12': [11],
			},
		},
	},
} as const;

describe('world JSON import/export', () => {
	test('round-trips ramp_demo.json without changing canonical structure', () => {
		const fixture = readWorldFixture('ramp_demo.json');
		expect(worldJsonFromDocument(documentFromWorldJson(fixture))).toEqual({
			...fixture,
			metadata: {
				spawn: null,
				target: null,
				direction: null,
				mirror_enabled: false,
			},
		});
	});

	test('round-trips a multi-object world without changing canonical structure', () => {
		expect(worldJsonFromDocument(documentFromWorldJson(multiObjectFixture))).toEqual(multiObjectFixture);
	});

	test('preserves custom connectivity instead of rebuilding geometric adjacency', () => {
		const fixture = {
			grid_width: 8,
			grid_height: 6,
			metadata: {
				spawn: null,
				target: null,
				direction: null,
				mirror_enabled: false,
			},
			objects: {
				robot: {
					indices: [9, 10, 11],
					types: [1, 1, 1],
					neighbors: {
						'9': [11],
						'10': [],
						'11': [9],
					},
				},
			},
		};

		const imported = documentFromWorldJson(fixture);
		expect(imported.objects[0]?.connectivity).toEqual({
			'0:0': [{ x: 2, y: 0 }],
			'1:0': [],
			'2:0': [{ x: 0, y: 0 }],
		});
		const restoredDraft = JSON.parse(JSON.stringify(imported));
		expect(worldJsonFromDocument(restoredDraft)).toEqual(fixture);
	});

	test('regenerates four-neighbor connectivity after a geometry edit', () => {
		const fixture = {
			grid_width: 8,
			grid_height: 6,
			objects: {
				robot: {
					indices: [9, 10, 11],
					types: [1, 1, 1],
					neighbors: { '9': [11], '10': [], '11': [9] },
				},
			},
		};
		const imported = documentFromWorldJson(fixture);
		const edited = setVoxelOnObject(imported, imported.objects[0]!.id, 1, 2, 1);
		expect(edited.objects[0]!.connectivity).toBeNull();
		expect(worldJsonFromDocument(edited).objects.robot?.neighbors).toEqual({
			'9': [10, 17],
			'10': [9, 11],
			'11': [10],
			'17': [9],
		});
	});

	test('rejects unknown voxel type IDs with the object and field location', () => {
		const fixture = {
			grid_width: 4,
			grid_height: 4,
			objects: {
				robot: {
					indices: [0],
					types: [99],
					neighbors: { '0': [] },
				},
			},
		};

		expect(() => documentFromWorldJson(fixture)).toThrow(
			/object 'robot'\.types\[0\] has unknown voxel type ID 99/,
		);
	});

	test('rejects zero-sized grids before converting flat indices', () => {
		const fixture = {
			grid_width: 0,
			grid_height: 4,
			objects: {},
		};

		expect(() => documentFromWorldJson(fixture)).toThrow(/grid_width must be at least 1/);
	});

	test('refuses to export an out-of-bounds object even when called without UI validation', () => {
		const document = documentFromWorldJson(multiObjectFixture);
		document.objects[0]!.origin.x = -1;
		expect(() => worldJsonFromDocument(document)).toThrow(/out-of-bounds voxel/);
	});

	test('refuses to export duplicate object names instead of overwriting a key', () => {
		const document = documentFromWorldJson(multiObjectFixture);
		document.objects[1]!.name = document.objects[0]!.name;
		expect(() => worldJsonFromDocument(document)).toThrow(/duplicate object name/);
	});

	test('rejects indices outside the declared grid', () => {
		const fixture = {
			grid_width: 4,
			grid_height: 4,
			objects: {
				robot: {
					indices: [16],
					types: [1],
					neighbors: { '16': [] },
				},
			},
		};

		expect(() => documentFromWorldJson(fixture)).toThrow(
			/object 'robot' index 16 is outside the declared 4x4 grid/,
		);
	});

	test('rejects duplicate indices', () => {
		const fixture = {
			grid_width: 4,
			grid_height: 4,
			objects: {
				robot: {
					indices: [1, 1],
					types: [1, 2],
					neighbors: { '1': [] },
				},
			},
		};

		expect(() => documentFromWorldJson(fixture)).toThrow(/object 'robot' has duplicate indices/);
	});

	test('imports missing metadata using defaults', () => {
		const fixture = {
			grid_width: 4,
			grid_height: 4,
			objects: {
				ground: {
					indices: [0],
					types: [5],
					neighbors: { '0': [] },
				},
			},
		};
		expect(documentFromWorldJson(fixture).metadata).toEqual({
			spawn: null,
			target: null,
			direction: null,
			mirrorEnabled: false,
		});
	});

	test('sanitizes malformed metadata while importing geometry', () => {
		const fixture = {
			grid_width: 4,
			grid_height: 4,
			metadata: {
				spawn: { x: '3', y: 0 },
				target: { x: 2.5, y: 2 },
				direction: 'up',
				mirror_enabled: 'yes',
			},
			objects: {
				ground: {
					indices: [0],
					types: [5],
					neighbors: { '0': [] },
				},
			},
		};
		const document = documentFromWorldJson(fixture);
		expect(document.objects).toHaveLength(1);
		expect(document.metadata).toEqual({
			spawn: null,
			target: null,
			direction: null,
			mirrorEnabled: false,
		});
	});

	test('import rejects invalid 2x1 slope pairings', () => {
		const invalidFixture = {
			grid_width: 4,
			grid_height: 4,
			metadata: {
				spawn: null,
				target: null,
				direction: null,
				mirror_enabled: false,
			},
			objects: {
				ground: {
					indices: [5],
					types: [12],
					neighbors: {
						'5': [],
					},
				},
			},
		};
		expect(() => documentFromWorldJson(invalidFixture)).toThrow(/invalid 2x1 slope pairing/i);
	});

	test('import rejects invalid 3x1 slope triplets', () => {
		const invalidFixture = {
			grid_width: 6,
			grid_height: 4,
			metadata: {
				spawn: null,
				target: null,
				direction: null,
				mirror_enabled: false,
			},
			objects: {
				ground: {
					indices: [5],
					types: [21],
					neighbors: {
						'5': [],
					},
				},
			},
		};
		expect(() => documentFromWorldJson(invalidFixture)).toThrow(/invalid 3x1 slope triplet/i);
	});
});
