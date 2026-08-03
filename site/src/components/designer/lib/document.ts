import {
	DEFAULT_GRID_HEIGHT,
	DEFAULT_GRID_WIDTH,
	MAX_GRID_DIM,
	MULTI_CELL_SLOPE_TYPES,
	SLOPE_2X1_HEAVY_TYPES,
	SLOPE_2X1_MATE_OFFSET,
	SLOPE_2X1_MATE_TYPE,
	SLOPE_2X1_TYPES,
	SLOPE_3X1_HEAVY_ANCHOR_TYPE,
	SLOPE_3X1_NEIGHBOR_RULES,
	SLOPE_3X1_SEGMENTS_FROM_HEAVY,
	SLOPE_3X1_TYPES,
	SLOPE_MIRROR_X_MAP,
	SLOPE_ROTATE_CW_MAP,
} from '../constants';
import type {
	DocumentPayload,
	EditorConnectivity,
	EditorObject,
	GridPoint,
	VoxelType,
	WorldDirection,
	WorldMetadata,
	WorldPoint,
} from '../types';
import { objectCells, worldCellForLocal } from './geometry';

function makeObjectId() {
	if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
		return `obj-${crypto.randomUUID()}`;
	}

	return `obj-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createDefaultMetadata(): WorldMetadata {
	return {
		spawn: null,
		target: null,
		direction: null,
		mirrorEnabled: false,
	};
}

function cloneWorldPoint(point: WorldPoint | null) {
	return point ? { ...point } : null;
}

export function cloneMetadata(metadata: WorldMetadata): WorldMetadata {
	return {
		spawn: cloneWorldPoint(metadata.spawn),
		target: cloneWorldPoint(metadata.target),
		direction: metadata.direction,
		mirrorEnabled: metadata.mirrorEnabled,
	};
}

export function normalizeWorldPoint(point: WorldPoint): WorldPoint {
	return {
		x: Math.floor(point.x),
		y: Math.floor(point.y),
	};
}

export function mirrorPointAcrossVerticalAxis(point: WorldPoint, gridWidth: number): WorldPoint {
	return {
		x: gridWidth - 1 - point.x,
		y: point.y,
	};
}

export function mirrorDirectionAcrossVerticalAxis(direction: WorldDirection): WorldDirection {
	if (direction === 'east') {
		return 'west';
	}
	if (direction === 'west') {
		return 'east';
	}
	return direction;
}

function clampGridDimension(value: number, fallback: number) {
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(MAX_GRID_DIM, Math.max(1, Math.floor(value)));
}

export function createEmptyDocument(width = DEFAULT_GRID_WIDTH, height = DEFAULT_GRID_HEIGHT): DocumentPayload {
	return {
		gridWidth: clampGridDimension(width, DEFAULT_GRID_WIDTH),
		gridHeight: clampGridDimension(height, DEFAULT_GRID_HEIGHT),
		objects: [],
		metadata: createDefaultMetadata(),
	};
}

export function cloneDocument(document: DocumentPayload): DocumentPayload {
	return {
		gridWidth: document.gridWidth,
		gridHeight: document.gridHeight,
		objects: document.objects.map((object) => ({
			...object,
			origin: { ...object.origin },
			voxels: object.voxels.map((voxel) => ({ ...voxel })),
			connectivity: cloneConnectivity(object.connectivity),
		})),
		metadata: cloneMetadata(document.metadata ?? createDefaultMetadata()),
	};
}

function cloneConnectivity(connectivity: EditorConnectivity | null | undefined) {
	if (!connectivity) {
		return null;
	}

	return Object.fromEntries(
		Object.entries(connectivity).map(([key, neighbors]) => [
			key,
			neighbors.map((neighbor) => ({ ...neighbor })),
		]),
	);
}

function pointKey(point: GridPoint) {
	return `${point.x}:${point.y}`;
}

function mapConnectivity(
	connectivity: EditorConnectivity | null,
	transform: (point: GridPoint) => GridPoint,
) {
	if (!connectivity) {
		return null;
	}

	return Object.fromEntries(
		Object.entries(connectivity).map(([key, neighbors]) => {
			const [x, y] = key.split(':').map(Number);
			const transformed = transform({ x, y });
			return [pointKey(transformed), neighbors.map(transform)];
		}),
	);
}

function remapSlopeType(type: VoxelType, mapping: Record<VoxelType, VoxelType>) {
	return mapping[type] ?? type;
}

function voxelIndexAt(object: EditorObject, localX: number, localY: number) {
	return object.voxels.findIndex((voxel) => voxel.x === localX && voxel.y === localY);
}

function setOrInsertVoxel(object: EditorObject, localX: number, localY: number, voxelType: VoxelType) {
	const voxelIndex = voxelIndexAt(object, localX, localY);
	if (voxelIndex >= 0) {
		object.voxels[voxelIndex]!.type = voxelType;
		return;
	}
	object.voxels.push({ x: localX, y: localY, type: voxelType });
}

function canonicalSlopePaintType(voxelType: VoxelType): VoxelType {
	if (SLOPE_2X1_TYPES.has(voxelType)) {
		return SLOPE_2X1_HEAVY_TYPES.has(voxelType)
			? voxelType
			: (SLOPE_2X1_MATE_TYPE[voxelType] ?? voxelType);
	}
	if (SLOPE_3X1_TYPES.has(voxelType)) {
		return SLOPE_3X1_HEAVY_ANCHOR_TYPE[voxelType] ?? voxelType;
	}
	return voxelType;
}

function multiCellSlopePattern(localX: number, localY: number, voxelType: VoxelType) {
	if (SLOPE_2X1_TYPES.has(voxelType)) {
		const paintType = canonicalSlopePaintType(voxelType);
		const mateType = SLOPE_2X1_MATE_TYPE[paintType];
		const mateOffset = SLOPE_2X1_MATE_OFFSET[paintType];
		if (typeof mateType !== 'number' || !mateOffset) {
			return null;
		}
		return [
			{ x: localX, y: localY, type: paintType },
			{ x: localX + mateOffset[0], y: localY + mateOffset[1], type: mateType },
		];
	}

	if (SLOPE_3X1_TYPES.has(voxelType)) {
		const paintType = canonicalSlopePaintType(voxelType);
		const segments = SLOPE_3X1_SEGMENTS_FROM_HEAVY[paintType];
		if (!segments) {
			return null;
		}
		return segments.map(([dx, dy, type]) => ({
			x: localX + dx,
			y: localY + dy,
			type,
		}));
	}

	return null;
}

function multiCellSlopeCoordsForSource(localX: number, localY: number, voxelType: VoxelType) {
	if (SLOPE_2X1_TYPES.has(voxelType)) {
		const mateType = SLOPE_2X1_MATE_TYPE[voxelType];
		const mateOffset = SLOPE_2X1_MATE_OFFSET[voxelType];
		if (typeof mateType !== 'number' || !mateOffset) {
			return [{ x: localX, y: localY }];
		}
		return [
			{ x: localX, y: localY },
			{ x: localX + mateOffset[0], y: localY + mateOffset[1] },
		];
	}

	if (SLOPE_3X1_TYPES.has(voxelType)) {
		const rules = SLOPE_3X1_NEIGHBOR_RULES[voxelType] ?? [];
		return [
			{ x: localX, y: localY },
			...rules.map(([dx, dy]) => ({ x: localX + dx, y: localY + dy })),
		];
	}

	return [{ x: localX, y: localY }];
}

function removeMultiCellSlopeAt(object: EditorObject, localX: number, localY: number) {
	const sourceIndex = voxelIndexAt(object, localX, localY);
	if (sourceIndex < 0) {
		return;
	}
	const source = object.voxels[sourceIndex];
	if (!source || !MULTI_CELL_SLOPE_TYPES.has(source.type)) {
		return;
	}
	const coords = multiCellSlopeCoordsForSource(localX, localY, source.type);
	const coordSet = new Set(coords.map((coord) => `${coord.x}:${coord.y}`));
	object.voxels = object.voxels.filter((voxel) => {
		return !coordSet.has(`${voxel.x}:${voxel.y}`);
	});
}

export function objectHasSlope2Cells(object: EditorObject) {
	return object.voxels.some((voxel) => SLOPE_2X1_TYPES.has(voxel.type));
}

export function objectHasMultiCellSlopeCells(object: EditorObject) {
	return object.voxels.some((voxel) => MULTI_CELL_SLOPE_TYPES.has(voxel.type));
}

export function normalizeObject(object: EditorObject) {
	if (!object.voxels.length) {
		return object;
	}

	const minX = Math.min(...object.voxels.map((voxel) => voxel.x));
	const minY = Math.min(...object.voxels.map((voxel) => voxel.y));

	if (minX === 0 && minY === 0) {
		return object;
	}

	for (const voxel of object.voxels) {
		voxel.x -= minX;
		voxel.y -= minY;
	}

	object.origin.x += minX;
	object.origin.y += minY;
	object.connectivity = mapConnectivity(object.connectivity, (point) => ({
		x: point.x - minX,
		y: point.y - minY,
	}));

	return object;
}

export function nextObjectName(document: DocumentPayload) {
	let counter = document.objects.length + 1;
	let candidate = `object_${counter}`;

	while (document.objects.some((object) => object.name === candidate)) {
		counter += 1;
		candidate = `object_${counter}`;
	}

	return candidate;
}

function makeUniqueCopyName(document: DocumentPayload, baseName: string) {
	const base = `${baseName}_copy`;
	let candidate = base;
	let counter = 2;

	while (document.objects.some((object) => object.name === candidate)) {
		candidate = `${base}_${counter}`;
		counter += 1;
	}

	return candidate;
}

export function ensureValidSelection(document: DocumentPayload, selectedObjectId: string | null) {
	if (selectedObjectId && document.objects.some((object) => object.id === selectedObjectId)) {
		return selectedObjectId;
	}

	return document.objects[0]?.id ?? null;
}

export function createObjectAt(
	document: DocumentPayload,
	worldX: number,
	worldY: number,
	voxelType: VoxelType,
) {
	const nextDocument = cloneDocument(document);
	const localVoxels: EditorObject['voxels'] = [];
	const slopePattern = multiCellSlopePattern(0, 0, voxelType);
	if (slopePattern) {
		localVoxels.push(...slopePattern);
	} else {
		localVoxels.push({ x: 0, y: 0, type: voxelType });
	}
	const object: EditorObject = {
		id: makeObjectId(),
		name: nextObjectName(nextDocument),
		origin: { x: worldX, y: worldY },
		voxels: localVoxels,
		connectivity: null,
		visible: true,
		locked: false,
	};
	normalizeObject(object);

	nextDocument.objects.push(object);

	return { document: nextDocument, objectId: object.id };
}

function findObjectById(document: DocumentPayload, objectId: string) {
	return document.objects.find((object) => object.id === objectId) ?? null;
}

export function findObjectAt(
	document: DocumentPayload,
	worldX: number,
	worldY: number,
	options?: { visibleOnly?: boolean },
) {
	const visibleOnly = options?.visibleOnly ?? false;

	return (
		document.objects.find((object) => {
			if (visibleOnly && !object.visible) {
				return false;
			}

			return objectCells(object).some((voxel) => {
				const world = worldCellForLocal(object, voxel);
				return world.x === worldX && world.y === worldY;
			});
		}) ?? null
	);
}

export function setVoxelOnObject(
	document: DocumentPayload,
	objectId: string,
	worldX: number,
	worldY: number,
	voxelType: VoxelType,
) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object) {
		return document;
	}

	const localX = worldX - object.origin.x;
	const localY = worldY - object.origin.y;
	const existingIndex = voxelIndexAt(object, localX, localY);
	const existing = existingIndex >= 0 ? object.voxels[existingIndex] : null;
	if (existing?.type === voxelType) {
		return document;
	}

	const slopePattern = multiCellSlopePattern(localX, localY, voxelType);
	if (slopePattern) {
		for (const segment of slopePattern) {
			removeMultiCellSlopeAt(object, segment.x, segment.y);
		}
		for (const segment of slopePattern) {
			setOrInsertVoxel(object, segment.x, segment.y, segment.type);
		}
	} else {
		if (existing && MULTI_CELL_SLOPE_TYPES.has(existing.type)) {
			removeMultiCellSlopeAt(object, localX, localY);
		}
		setOrInsertVoxel(object, localX, localY, voxelType);
	}

	// Geometry edits deliberately regenerate the entire object's four-neighbor
	// graph on export. This avoids retaining custom edges to removed cells.
	object.connectivity = null;
	normalizeObject(object);

	return nextDocument;
}

export function removeVoxelFromObject(
	document: DocumentPayload,
	objectId: string,
	worldX: number,
	worldY: number,
) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object) {
		return document;
	}

	const localX = worldX - object.origin.x;
	const localY = worldY - object.origin.y;
	const sourceIndex = voxelIndexAt(object, localX, localY);
	if (sourceIndex < 0) {
		return document;
	}
	const source = object.voxels[sourceIndex];
	if (!source) {
		return document;
	}

	if (MULTI_CELL_SLOPE_TYPES.has(source.type)) {
		removeMultiCellSlopeAt(object, localX, localY);
	} else {
		object.voxels = object.voxels.filter((voxel) => !(voxel.x === localX && voxel.y === localY));
	}
	object.connectivity = null;
	normalizeObject(object);

	if (!objectCells(object).length) {
		nextDocument.objects = nextDocument.objects.filter((candidate) => candidate.id !== object.id);
	}

	return nextDocument;
}

export function paintRect(
	document: DocumentPayload,
	objectId: string,
	start: GridPoint,
	end: GridPoint,
	voxelType: VoxelType,
) {
	if (MULTI_CELL_SLOPE_TYPES.has(voxelType)) {
		return document;
	}
	let nextDocument = document;
	const minX = Math.min(start.x, end.x);
	const maxX = Math.max(start.x, end.x);
	const minY = Math.min(start.y, end.y);
	const maxY = Math.max(start.y, end.y);

	for (let x = minX; x <= maxX; x += 1) {
		for (let y = minY; y <= maxY; y += 1) {
			nextDocument = setVoxelOnObject(nextDocument, objectId, x, y, voxelType);
		}
	}

	return nextDocument;
}

export function moveObject(document: DocumentPayload, objectId: string, deltaX: number, deltaY: number) {
	if (deltaX === 0 && deltaY === 0) {
		return document;
	}

	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object) {
		return document;
	}

	object.origin.x += deltaX;
	object.origin.y += deltaY;

	return nextDocument;
}

export function duplicateObject(document: DocumentPayload, objectId: string) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object) {
		return { document, objectId: null };
	}

	const duplicate: EditorObject = {
		...object,
		id: makeObjectId(),
		name: makeUniqueCopyName(nextDocument, object.name),
		origin: {
			x: object.origin.x + 1,
			y: object.origin.y + 1,
		},
		voxels: object.voxels.map((voxel) => ({ ...voxel })),
		connectivity: cloneConnectivity(object.connectivity),
	};

	nextDocument.objects.push(duplicate);

	return { document: nextDocument, objectId: duplicate.id };
}

export function rotateObject(document: DocumentPayload, objectId: string) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object || object.locked) {
		return document;
	}

	const cells = objectCells(object);

	if (!cells.length) {
		return document;
	}
	if (objectHasMultiCellSlopeCells(object)) {
		return document;
	}

	const maxX = Math.max(...cells.map((voxel) => voxel.x));
	object.voxels = cells.map((voxel) => ({
		x: voxel.y,
		y: maxX - voxel.x,
		type: remapSlopeType(voxel.type, SLOPE_ROTATE_CW_MAP),
	}));
	object.connectivity = mapConnectivity(object.connectivity, (point) => ({
		x: point.y,
		y: maxX - point.x,
	}));

	normalizeObject(object);

	return nextDocument;
}

export function mirrorObject(document: DocumentPayload, objectId: string) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object || object.locked) {
		return document;
	}

	const cells = objectCells(object);

	if (!cells.length) {
		return document;
	}

	const maxX = Math.max(...cells.map((voxel) => voxel.x));
	object.voxels = cells.map((voxel) => ({
		x: maxX - voxel.x,
		y: voxel.y,
		type: remapSlopeType(voxel.type, SLOPE_MIRROR_X_MAP),
	}));
	object.connectivity = mapConnectivity(object.connectivity, (point) => ({
		x: maxX - point.x,
		y: point.y,
	}));

	normalizeObject(object);

	return nextDocument;
}

export function deleteObject(document: DocumentPayload, objectId: string) {
	const nextDocument = cloneDocument(document);
	const nextObjects = nextDocument.objects.filter((object) => object.id !== objectId);

	if (nextObjects.length === nextDocument.objects.length) {
		return document;
	}

	nextDocument.objects = nextObjects;
	return nextDocument;
}

export function renameObject(document: DocumentPayload, objectId: string, name: string) {
	const trimmed = name.trim();

	if (!trimmed) {
		return document;
	}

	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object || object.name === trimmed) {
		return document;
	}

	object.name = trimmed;
	return nextDocument;
}

export function toggleObjectVisibility(document: DocumentPayload, objectId: string) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object) {
		return document;
	}

	object.visible = !object.visible;
	return nextDocument;
}

export function toggleObjectLocked(document: DocumentPayload, objectId: string) {
	const nextDocument = cloneDocument(document);
	const object = findObjectById(nextDocument, objectId);

	if (!object) {
		return document;
	}

	object.locked = !object.locked;
	return nextDocument;
}

export function setGridSize(document: DocumentPayload, width: number, height: number) {
	const nextWidth = clampGridDimension(width, document.gridWidth);
	const nextHeight = clampGridDimension(height, document.gridHeight);

	if (nextWidth === document.gridWidth && nextHeight === document.gridHeight) {
		return document;
	}

	return {
		...cloneDocument(document),
		gridWidth: nextWidth,
		gridHeight: nextHeight,
	};
}
