import type {
	DocumentPayload,
	EditorObject,
	WorldDirection,
	WorldJson,
	WorldJsonObject,
	WorldPoint,
} from '../types';
import {
	MAX_GRID_DIM,
	SLOPE_2X1_MATE_OFFSET,
	SLOPE_2X1_MATE_TYPE,
	SLOPE_2X1_TYPES,
	SLOPE_3X1_NEIGHBOR_RULES,
	SLOPE_3X1_TYPES,
	VOXEL_TYPE_IDS,
} from '../constants';
import { createDefaultMetadata } from './document';

function expectRecord(value: unknown, message: string): Record<string, unknown> {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		throw new Error(message);
	}

	return value as Record<string, unknown>;
}

function expectInteger(value: unknown, key: string) {
	if (!Number.isInteger(value)) {
		throw new Error(`${key} must be an integer`);
	}

	return Number(value);
}

function expectGridDimension(value: unknown, key: string) {
	const dimension = expectInteger(value, key);
	if (dimension < 1) {
		throw new Error(`${key} must be at least 1`);
	}
	if (dimension > MAX_GRID_DIM) {
		throw new Error(`${key} must be no greater than ${MAX_GRID_DIM}`);
	}
	return dimension;
}

function expectIntegerArray(value: unknown, key: string) {
	if (!Array.isArray(value) || !value.every((entry) => Number.isInteger(entry))) {
		throw new Error(`${key} must be an array of integers`);
	}

	return value.map((entry) => Number(entry));
}

function expectVoxelType(value: unknown, key: string): EditorObject['voxels'][number]['type'] {
	const type = expectInteger(value, key);
	if (!VOXEL_TYPE_IDS.has(type)) {
		throw new Error(`${key} has unknown voxel type ID ${type}`);
	}

	return type as EditorObject['voxels'][number]['type'];
}

const DIRECTION_VALUES: WorldDirection[] = ['north', 'east', 'south', 'west'];

function parseWorldPoint(value: unknown): WorldPoint | null {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return null;
	}

	const record = value as Record<string, unknown>;
	if (!Number.isInteger(record.x) || !Number.isInteger(record.y)) {
		return null;
	}

	return {
		x: Number(record.x),
		y: Number(record.y),
	};
}

function parseDirection(value: unknown): WorldDirection | null {
	return typeof value === 'string' && DIRECTION_VALUES.includes(value as WorldDirection)
		? (value as WorldDirection)
		: null;
}

function metadataFromWorldJson(value: unknown): DocumentPayload['metadata'] {
	const defaults = createDefaultMetadata();
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return defaults;
	}

	const record = value as Record<string, unknown>;
	return {
		spawn: parseWorldPoint(record.spawn),
		target: parseWorldPoint(record.target),
		direction: parseDirection(record.direction),
		mirrorEnabled: record.mirror_enabled === true,
	};
}

function worldJsonMetadataFromDocument(document: DocumentPayload): WorldJson['metadata'] {
	return {
		spawn: document.metadata.spawn,
		target: document.metadata.target,
		direction: document.metadata.direction,
		mirror_enabled: document.metadata.mirrorEnabled,
	};
}

function objectFromWorldEntry(
	name: string,
	objectId: string,
	rawObject: unknown,
	gridWidth: number,
	gridHeight: number,
): EditorObject {
	const objectRecord = expectRecord(rawObject, `object '${name}' must be an object`);
	const indices = expectIntegerArray(objectRecord.indices, `object '${name}'.indices`);
	if (!Array.isArray(objectRecord.types)) {
		throw new Error(`object '${name}'.types must be an array of voxel type IDs`);
	}
	const types = objectRecord.types.map((type, position) =>
		expectVoxelType(type, `object '${name}'.types[${position}]`),
	);
	const neighbors = expectRecord(objectRecord.neighbors, `object '${name}'.neighbors must be an object`);

	if (indices.length !== types.length) {
		throw new Error(`object '${name}' has mismatched indices and types`);
	}

	if (new Set(indices).size !== indices.length) {
		throw new Error(`object '${name}' has duplicate indices`);
	}
	if (indices.length !== Object.keys(neighbors).length) {
		throw new Error(`object '${name}' has mismatched indices and neighbors`);
	}

	const indexSet = new Set(indices);

	const voxels = indices.map((index, position) => {
		if (index < 0 || index >= gridWidth * gridHeight) {
			throw new Error(
				`object '${name}' index ${index} is outside the declared ${gridWidth}x${gridHeight} grid`,
			);
		}
		const neighborEntry = neighbors[String(index)];
		const neighborIndices = expectIntegerArray(neighborEntry, `object '${name}'.neighbors['${index}']`);
		for (const neighborIndex of neighborIndices) {
			if (!indexSet.has(neighborIndex)) {
				throw new Error(
					`object '${name}' index ${index} has unknown neighbor ${neighborIndex}`,
				);
			}
		}

		return {
			index,
			type: types[position]!,
			x: index % gridWidth,
			y: Math.floor(index / gridWidth),
			neighborIndices,
		};
	});
	validateMultiCellSlopePairs(name, voxels);

	if (!voxels.length) {
		throw new Error(`object '${name}' has no voxels`);
	}

	const minX = Math.min(...voxels.map((voxel) => voxel.x));
	const minY = Math.min(...voxels.map((voxel) => voxel.y));
	const localPointByIndex = new Map(
		voxels.map((voxel) => [
			voxel.index,
			{ x: voxel.x - minX, y: voxel.y - minY },
		]),
	);
	const connectivity = Object.fromEntries(
		voxels.map((voxel) => {
			const localPoint = localPointByIndex.get(voxel.index)!;
			return [
				`${localPoint.x}:${localPoint.y}`,
				voxel.neighborIndices.map((neighborIndex) => ({
					...localPointByIndex.get(neighborIndex)!,
				})),
			];
		}),
	);

	return {
		id: objectId,
		name,
		origin: { x: minX, y: minY },
		voxels: voxels.map((voxel) => ({
			x: voxel.x - minX,
			y: voxel.y - minY,
			type: voxel.type,
		})),
		connectivity,
		visible: true,
		locked: false,
	};
}

function validateMultiCellSlopePairs(
	name: string,
	voxels: Array<{ x: number; y: number; type: EditorObject['voxels'][number]['type'] }>,
) {
	const voxelByPoint = new Map(voxels.map((voxel) => [`${voxel.x}:${voxel.y}`, voxel]));
	for (const voxel of voxels) {
		if (!SLOPE_2X1_TYPES.has(voxel.type)) {
			continue;
		}
		const mateType = SLOPE_2X1_MATE_TYPE[voxel.type];
		const mateOffset = SLOPE_2X1_MATE_OFFSET[voxel.type];
		if (typeof mateType !== 'number' || !mateOffset) {
			continue;
		}
		const mateX = voxel.x + mateOffset[0];
		const mateY = voxel.y + mateOffset[1];
		const mate = voxelByPoint.get(`${mateX}:${mateY}`);
		if (!mate || mate.type !== mateType) {
			throw new Error(
				`object '${name}' has invalid 2x1 slope pairing at (${voxel.x}, ${voxel.y})`,
			);
		}
	}
	for (const voxel of voxels) {
		if (!SLOPE_3X1_TYPES.has(voxel.type)) {
			continue;
		}
		const neighborRules = SLOPE_3X1_NEIGHBOR_RULES[voxel.type] ?? [];
		for (const [dx, dy, expectedType] of neighborRules) {
			const mateX = voxel.x + dx;
			const mateY = voxel.y + dy;
			const mate = voxelByPoint.get(`${mateX}:${mateY}`);
			if (!mate || mate.type !== expectedType) {
				throw new Error(
					`object '${name}' has invalid 3x1 slope triplet at (${voxel.x}, ${voxel.y})`,
				);
			}
		}
	}
}

export function documentFromWorldJson(value: unknown): DocumentPayload {
	const record = expectRecord(value, 'world JSON must be an object');
	const gridWidth = expectGridDimension(record.grid_width, 'grid_width');
	const gridHeight = expectGridDimension(record.grid_height, 'grid_height');
	const objectsRecord = expectRecord(record.objects, 'objects');

	const objects = Object.entries(objectsRecord)
		.sort(([left], [right]) => left.localeCompare(right))
		.map(([name, rawObject], index) =>
			objectFromWorldEntry(name, `obj-${index + 1}`, rawObject, gridWidth, gridHeight),
		);

	return {
		gridWidth,
		gridHeight,
		objects,
		metadata: metadataFromWorldJson(record.metadata),
	};
}

function worldObjectFromDocument(object: EditorObject, gridWidth: number): WorldJsonObject {
	const occupied = object.voxels.map((voxel) => ({
		worldX: object.origin.x + voxel.x,
		worldY: object.origin.y + voxel.y,
		localX: voxel.x,
		localY: voxel.y,
		type: voxel.type,
	}));

	const indexByLocal = new Map<string, number>();
	for (const voxel of occupied) {
		indexByLocal.set(`${voxel.localX}:${voxel.localY}`, voxel.worldY * gridWidth + voxel.worldX);
	}

	const ordered = occupied
		.map((voxel) => ({
			...voxel,
			index: voxel.worldY * gridWidth + voxel.worldX,
		}))
		.sort((left, right) => left.index - right.index);

	const neighborSets = new Map<number, Set<number>>();
	for (const voxel of ordered) {
		const neighborIndices = new Set<number>();
		if (object.connectivity) {
			const localKey = `${voxel.localX}:${voxel.localY}`;
			const customNeighbors = object.connectivity[localKey];
			if (!customNeighbors) {
				throw new Error(
					`object '${object.name}' is missing connectivity for local cell (${voxel.localX}, ${voxel.localY})`,
				);
			}
			for (const neighbor of customNeighbors) {
				const candidate = indexByLocal.get(`${neighbor.x}:${neighbor.y}`);
				if (typeof candidate !== 'number') {
					throw new Error(
						`object '${object.name}' has connectivity to missing local cell (${neighbor.x}, ${neighbor.y})`,
					);
				}
				neighborIndices.add(candidate);
			}
		} else {
			for (const [dx, dy] of [
				[-1, 0],
				[1, 0],
				[0, -1],
				[0, 1],
			]) {
				const candidate = indexByLocal.get(`${voxel.localX + dx}:${voxel.localY + dy}`);
				if (typeof candidate === 'number') {
					neighborIndices.add(candidate);
				}
			}
		}
		neighborSets.set(voxel.index, neighborIndices);
	}

	const neighbors: Record<string, number[]> = {};
	for (const voxel of ordered) {
		neighbors[String(voxel.index)] = [...(neighborSets.get(voxel.index) ?? [])].sort(
			(left, right) => left - right,
		);
	}

	return {
		indices: ordered.map((voxel) => voxel.index),
		types: ordered.map((voxel) => voxel.type),
		neighbors,
	};
}

export function worldJsonFromDocument(document: DocumentPayload): WorldJson {
	if (!Number.isInteger(document.gridWidth) || document.gridWidth < 1) {
		throw new Error('grid width must be a positive integer before export');
	}
	if (!Number.isInteger(document.gridHeight) || document.gridHeight < 1) {
		throw new Error('grid height must be a positive integer before export');
	}

	const names = new Set<string>();
	for (const object of document.objects) {
		if (!object.name.trim()) {
			throw new Error('every object must have a non-empty name before export');
		}
		if (names.has(object.name)) {
			throw new Error(`duplicate object name '${object.name}' cannot be exported`);
		}
		names.add(object.name);

		const occupied = new Set<string>();
		for (const voxel of object.voxels) {
			if (!VOXEL_TYPE_IDS.has(voxel.type)) {
				throw new Error(`object '${object.name}' has unknown voxel type ID ${voxel.type}`);
			}
			const worldX = object.origin.x + voxel.x;
			const worldY = object.origin.y + voxel.y;
			if (
				!Number.isInteger(worldX) ||
				!Number.isInteger(worldY) ||
				worldX < 0 ||
				worldX >= document.gridWidth ||
				worldY < 0 ||
				worldY >= document.gridHeight
			) {
				throw new Error(
					`object '${object.name}' has an out-of-bounds voxel at (${worldX}, ${worldY})`,
				);
			}
			const key = `${voxel.x}:${voxel.y}`;
			if (occupied.has(key)) {
				throw new Error(`object '${object.name}' has duplicate local voxel (${voxel.x}, ${voxel.y})`);
			}
			occupied.add(key);
		}
	}

	const objects = Object.fromEntries(
		[...document.objects]
			.sort((left, right) => left.name.localeCompare(right.name))
			.map((object) => [object.name, worldObjectFromDocument(object, document.gridWidth)]),
	);

	return {
		grid_width: document.gridWidth,
		grid_height: document.gridHeight,
		objects,
		metadata: worldJsonMetadataFromDocument(document),
	};
}

export function stringifyWorldJson(document: DocumentPayload) {
	return `${JSON.stringify(worldJsonFromDocument(document), null, 4)}\n`;
}

export async function importWorldFile(file: File) {
	const text = await file.text();
	return documentFromWorldJson(JSON.parse(text) as unknown);
}

export function exportWorldFile(doc: DocumentPayload, fileName = 'evogym-world.json') {
	const payload = stringifyWorldJson(doc);
	const blob = new Blob([payload], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const link = document.createElement('a');
	link.href = url;
	link.download = fileName;
	link.click();
	URL.revokeObjectURL(url);
}
