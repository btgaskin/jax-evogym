import { CELL_VERTEX_OFFSETS, FIXED } from '../constants';
import type { EditorObject, GridPoint, VoxelCell, VoxelType } from '../types';

const EDGE_KEY_SCALE = 6;

export interface WorldCell extends GridPoint {
	type: VoxelType;
}

export interface GeometryPolygon {
	cell: WorldCell;
	vertices: [number, number][];
	surfaceMask: boolean[];
}

export function objectCells(object: EditorObject) {
	return object.voxels;
}

export function worldCellForLocal(object: EditorObject, voxel: VoxelCell): WorldCell {
	return {
		x: object.origin.x + voxel.x,
		y: object.origin.y + voxel.y,
		type: voxel.type,
	};
}

export function cellVertexOffsets(type: VoxelType) {
	return CELL_VERTEX_OFFSETS[type] ?? CELL_VERTEX_OFFSETS[FIXED];
}

export function cellVerticesWorld(cell: WorldCell) {
	return cellVertexOffsets(cell.type).map(([dx, dy]) => [cell.x + dx, cell.y + dy] as [number, number]);
}

export function canonicalEdgeKey(pointA: [number, number], pointB: [number, number]) {
	const [ax, ay] = pointA.map((value) => Math.round(value * EDGE_KEY_SCALE));
	const [bx, by] = pointB.map((value) => Math.round(value * EDGE_KEY_SCALE));

	if (ax < bx || (ax === bx && ay <= by)) {
		return `${ax}:${ay}|${bx}:${by}`;
	}

	return `${bx}:${by}|${ax}:${ay}`;
}

export function objectGeometry(object: EditorObject): GeometryPolygon[] {
	const cells = objectCells(object).map((voxel) => worldCellForLocal(object, voxel));
	const edgeCounts = new Map<string, number>();

	const polygons = cells.map((cell) => {
		const vertices = cellVerticesWorld(cell);

		for (let index = 0; index < vertices.length; index += 1) {
			const nextIndex = (index + 1) % vertices.length;
			const key = canonicalEdgeKey(vertices[nextIndex], vertices[index]);
			edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
		}

		return { cell, vertices };
	});

	return polygons.map(({ cell, vertices }) => ({
		cell,
		vertices,
		surfaceMask: vertices.map((_, index) => {
			const nextIndex = (index + 1) % vertices.length;
			const key = canonicalEdgeKey(vertices[nextIndex], vertices[index]);
			return edgeCounts.get(key) === 1;
		}),
	}));
}

export function objectBounds(object: EditorObject) {
	const cells = objectCells(object);

	if (!cells.length) {
		return null;
	}

	const xs = cells.map((voxel) => object.origin.x + voxel.x);
	const ys = cells.map((voxel) => object.origin.y + voxel.y);

	return {
		minX: Math.min(...xs),
		minY: Math.min(...ys),
		maxX: Math.max(...xs),
		maxY: Math.max(...ys),
	};
}
