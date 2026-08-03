import {
	CANVAS_PADDING,
	DEFAULT_CELL_SIZE,
	EDGE_WIDTH_PX,
	GRID_MAJOR_EVERY,
	SLOPE_TYPES,
	VOXEL_DEFINITION_BY_TYPE,
	getDesignThemeTokens,
	type DesignThemeTokens,
	type ThemeMode,
} from '../constants';
import type { DocumentPayload, RectPreview, ViewState, VoxelType, WorldDirection, WorldPoint } from '../types';
import { mirrorDirectionAcrossVerticalAxis, mirrorPointAcrossVerticalAxis } from '../lib/document';
import { objectBounds, objectGeometry } from '../lib/geometry';

export const MIN_VIEW_SCALE = 0.18;
export const MAX_VIEW_SCALE = 2.4;

export interface DrawFrameInput {
	ctx: CanvasRenderingContext2D;
	document: DocumentPayload;
	selectedObjectId: string | null;
	view: ViewState;
	width: number;
	height: number;
	rectPreview: RectPreview | null;
	theme: ThemeMode;
}

type DesignRenderTokens = DesignThemeTokens['render'];

function pixelAligned(position: number) {
	return Math.round(position) + 0.5;
}

export function gridPointToCanvas(x: number, y: number, view: ViewState, height: number) {
	const cellSize = DEFAULT_CELL_SIZE * view.scale;
	return {
		x: CANVAS_PADDING + view.offsetX + x * cellSize,
		y: height - CANVAS_PADDING + view.offsetY - y * cellSize,
	};
}

export function gridToCanvas(x: number, y: number, view: ViewState, height: number) {
	const point = gridPointToCanvas(x, y + 1, view, height);
	return {
		x: point.x,
		y: point.y,
		size: DEFAULT_CELL_SIZE * view.scale,
	};
}

export function canvasToGrid(clientX: number, clientY: number, rect: DOMRect, view: ViewState, height: number) {
	const x = clientX - rect.left;
	const y = clientY - rect.top;
	const cellSize = DEFAULT_CELL_SIZE * view.scale;

	return {
		x: Math.floor((x - CANVAS_PADDING - view.offsetX) / cellSize),
		y: Math.floor((height - CANVAS_PADDING + view.offsetY - y) / cellSize),
	};
}

export function getFittedView(document: DocumentPayload, width: number, height: number): ViewState {
	const widthScale = Math.max(1, width - CANVAS_PADDING * 2 - 8) / Math.max(1, document.gridWidth * DEFAULT_CELL_SIZE);
	const heightScale =
		Math.max(1, height - CANVAS_PADDING * 2 - 8) / Math.max(1, document.gridHeight * DEFAULT_CELL_SIZE);

	return {
		scale: Math.min(MAX_VIEW_SCALE, Math.max(MIN_VIEW_SCALE, Math.min(widthScale, heightScale))),
		offsetX: 0,
		offsetY: 0,
	};
}

function voxelFill(ctx: CanvasRenderingContext2D, type: number, x: number, y: number, size: number) {
	const definition = VOXEL_DEFINITION_BY_TYPE[type as keyof typeof VOXEL_DEFINITION_BY_TYPE];
	if (!definition) {
		return '#c5cbc4';
	}

	if (definition.kind === 'solid') {
		return definition.hex ?? '#c5cbc4';
	}

	const gradient = ctx.createLinearGradient(x, y, x, y + size);
	gradient.addColorStop(0, definition.start ?? '#ffffff');
	gradient.addColorStop(1, definition.end ?? '#000000');
	return gradient;
}

function drawCell(
	ctx: CanvasRenderingContext2D,
	type: number,
	vertices: { x: number; y: number }[],
	x: number,
	y: number,
	size: number,
) {
	ctx.fillStyle = voxelFill(ctx, type, x, y, size);

	if (!SLOPE_TYPES.has(type as VoxelType)) {
		ctx.fillRect(x, y, size, size);
		return;
	}

	ctx.beginPath();
	ctx.moveTo(vertices[0].x, vertices[0].y);
	for (let index = 1; index < vertices.length; index += 1) {
		ctx.lineTo(vertices[index].x, vertices[index].y);
	}
	ctx.closePath();
	ctx.fill();
}

function drawMarker(
	ctx: CanvasRenderingContext2D,
	view: ViewState,
	height: number,
	point: WorldPoint,
	label: string,
	color: string,
	fillColor: string,
	preview = false,
) {
	const cell = gridToCanvas(point.x, point.y, view, height);
	const centerX = cell.x + cell.size / 2;
	const centerY = cell.y + cell.size / 2;
	const radius = Math.max(6, cell.size * 0.24);

	ctx.save();
	if (preview) {
		ctx.globalAlpha = 0.56;
		ctx.setLineDash([5, 4]);
	}
	ctx.fillStyle = fillColor;
	ctx.strokeStyle = color;
	ctx.lineWidth = 2;
	ctx.beginPath();
	ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
	ctx.fill();
	ctx.stroke();
	ctx.restore();

	ctx.save();
	if (preview) {
		ctx.globalAlpha = 0.72;
	}
	ctx.fillStyle = color;
	ctx.font = `${Math.max(10, Math.round(cell.size * 0.42))}px ui-monospace, SFMono-Regular, Menlo, monospace`;
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	ctx.fillText(label, centerX, centerY + 0.5);
	ctx.restore();
}

function directionVector(direction: WorldDirection): [number, number] {
	if (direction === 'north') {
		return [0, -1];
	}
	if (direction === 'south') {
		return [0, 1];
	}
	if (direction === 'east') {
		return [1, 0];
	}
	return [-1, 0];
}

function drawDirectionArrow(
	ctx: CanvasRenderingContext2D,
	view: ViewState,
	height: number,
	origin: WorldPoint,
	direction: WorldDirection,
	color: string,
	preview = false,
) {
	const cell = gridToCanvas(origin.x, origin.y, view, height);
	const startX = cell.x + cell.size / 2;
	const startY = cell.y + cell.size / 2;
	const [dx, dy] = directionVector(direction);
	const length = Math.max(10, cell.size * 0.78);
	const endX = startX + dx * length;
	const endY = startY + dy * length;
	const headLength = Math.max(6, cell.size * 0.23);
	const angle = Math.atan2(dy, dx);

	ctx.save();
	ctx.strokeStyle = color;
	ctx.fillStyle = color;
	ctx.lineWidth = 2.5;
	ctx.lineCap = 'round';
	ctx.lineJoin = 'round';
	if (preview) {
		ctx.globalAlpha = 0.56;
		ctx.setLineDash([6, 4]);
	}
	ctx.beginPath();
	ctx.moveTo(startX, startY);
	ctx.lineTo(endX, endY);
	ctx.stroke();
	ctx.setLineDash([]);
	ctx.beginPath();
	ctx.moveTo(endX, endY);
	ctx.lineTo(
		endX - headLength * Math.cos(angle - Math.PI / 7),
		endY - headLength * Math.sin(angle - Math.PI / 7),
	);
	ctx.lineTo(
		endX - headLength * Math.cos(angle + Math.PI / 7),
		endY - headLength * Math.sin(angle + Math.PI / 7),
	);
	ctx.closePath();
	ctx.fill();
	ctx.restore();
}

function drawMetadataMarkers(
	ctx: CanvasRenderingContext2D,
	document: DocumentPayload,
	view: ViewState,
	height: number,
	renderTokens: DesignRenderTokens,
) {
	const metadata = document.metadata;
	if (metadata.spawn) {
		drawMarker(
			ctx,
			view,
			height,
			metadata.spawn,
			'S',
			renderTokens.marker_spawn,
			renderTokens.marker_fill,
		);
	}
	if (metadata.target) {
		drawMarker(
			ctx,
			view,
			height,
			metadata.target,
			'T',
			renderTokens.marker_target,
			renderTokens.marker_fill,
		);
	}
	if (metadata.spawn && metadata.direction) {
		drawDirectionArrow(
			ctx,
			view,
			height,
			metadata.spawn,
			metadata.direction,
			renderTokens.marker_direction,
		);
	}
	if (!metadata.mirrorEnabled) {
		return;
	}

	if (metadata.spawn) {
		const mirroredSpawn = mirrorPointAcrossVerticalAxis(metadata.spawn, document.gridWidth);
		if (mirroredSpawn.x !== metadata.spawn.x || mirroredSpawn.y !== metadata.spawn.y) {
			drawMarker(
				ctx,
				view,
				height,
				mirroredSpawn,
				'S',
				renderTokens.marker_spawn,
				renderTokens.marker_fill,
				true,
			);
		}
		if (metadata.direction) {
			drawDirectionArrow(
				ctx,
				view,
				height,
				mirroredSpawn,
				mirrorDirectionAcrossVerticalAxis(metadata.direction),
				renderTokens.marker_direction,
				true,
			);
		}
	}
	if (metadata.target) {
		const mirroredTarget = mirrorPointAcrossVerticalAxis(metadata.target, document.gridWidth);
		if (mirroredTarget.x !== metadata.target.x || mirroredTarget.y !== metadata.target.y) {
			drawMarker(
				ctx,
				view,
				height,
				mirroredTarget,
				'T',
				renderTokens.marker_target,
				renderTokens.marker_fill,
				true,
			);
		}
	}
}

export function drawGrid(
	ctx: CanvasRenderingContext2D,
	document: DocumentPayload,
	view: ViewState,
	width: number,
	height: number,
	renderTokens: DesignRenderTokens,
) {
	ctx.lineWidth = 1;

	for (let x = 0; x <= document.gridWidth; x += 1) {
		const start = gridPointToCanvas(x, 0, view, height);
		const end = gridPointToCanvas(x, document.gridHeight, view, height);
		ctx.beginPath();
		ctx.moveTo(pixelAligned(start.x), CANVAS_PADDING / 2);
		ctx.lineTo(pixelAligned(end.x), height - CANVAS_PADDING / 2);
		ctx.strokeStyle =
			x % GRID_MAJOR_EVERY === 0 ? renderTokens.grid_major : renderTokens.grid_minor;
		ctx.stroke();
	}

	for (let y = 0; y <= document.gridHeight; y += 1) {
		const start = gridPointToCanvas(0, y, view, height);
		ctx.beginPath();
		ctx.moveTo(CANVAS_PADDING / 2, pixelAligned(start.y));
		ctx.lineTo(width - CANVAS_PADDING / 2, pixelAligned(start.y));
		ctx.strokeStyle =
			y % GRID_MAJOR_EVERY === 0 ? renderTokens.grid_major : renderTokens.grid_minor;
		ctx.stroke();
	}
}

function drawSurfaceEdges(
	ctx: CanvasRenderingContext2D,
	document: DocumentPayload,
	view: ViewState,
	height: number,
	renderTokens: DesignRenderTokens,
	options?: {
		mirrored?: boolean;
		preview?: boolean;
	},
) {
	const mirrored = options?.mirrored ?? false;
	const preview = options?.preview ?? false;
	ctx.save();
	if (preview) {
		ctx.globalAlpha = 0.38;
		ctx.setLineDash([6, 4]);
	}
	ctx.strokeStyle = renderTokens.edge;
	ctx.lineWidth = EDGE_WIDTH_PX;

	for (const object of document.objects) {
		if (!object.visible) {
			continue;
		}

		for (const polygon of objectGeometry(object)) {
			const points = polygon.vertices.map(([x, y]) =>
				gridPointToCanvas(mirrored ? document.gridWidth - x : x, y, view, height),
			);
			const cellTopLeft = gridToCanvas(
				mirrored ? document.gridWidth - 1 - polygon.cell.x : polygon.cell.x,
				polygon.cell.y,
				view,
				height,
			);
			drawCell(ctx, polygon.cell.type, points, cellTopLeft.x, cellTopLeft.y, cellTopLeft.size);

			for (let index = 0; index < points.length; index += 1) {
				if (!polygon.surfaceMask[index]) {
					continue;
				}

				const nextIndex = (index + 1) % points.length;
				ctx.beginPath();
				ctx.moveTo(points[nextIndex].x, points[nextIndex].y);
				ctx.lineTo(points[index].x, points[index].y);
				ctx.stroke();
			}
		}
	}
	ctx.restore();
}

export function drawSelectionBox(
	ctx: CanvasRenderingContext2D,
	document: DocumentPayload,
	selectedObjectId: string | null,
	view: ViewState,
	height: number,
	renderTokens: DesignRenderTokens,
) {
	if (!selectedObjectId) {
		return;
	}

	const object = document.objects.find((candidate) => candidate.id === selectedObjectId);
	if (!object || !object.visible) {
		return;
	}

	const bounds = objectBounds(object);
	if (!bounds) {
		return;
	}

	const start = gridToCanvas(bounds.minX, bounds.maxY, view, height);
	const end = gridToCanvas(bounds.maxX + 1, bounds.minY - 1, view, height);

	ctx.save();
	ctx.strokeStyle = object.locked ? renderTokens.locked : renderTokens.selection;
	ctx.lineWidth = 3;
	if (object.locked) {
		ctx.setLineDash([8, 6]);
	}
	ctx.strokeRect(start.x - 3, start.y - 3, end.x - start.x + 6, end.y - start.y + 6);
	ctx.restore();
}

export function drawRectPreview(
	ctx: CanvasRenderingContext2D,
	view: ViewState,
	height: number,
	rectPreview: RectPreview | null,
	renderTokens: DesignRenderTokens,
) {
	if (!rectPreview) {
		return;
	}

	const minX = Math.min(rectPreview.start.x, rectPreview.current.x);
	const maxX = Math.max(rectPreview.start.x, rectPreview.current.x);
	const minY = Math.min(rectPreview.start.y, rectPreview.current.y);
	const maxY = Math.max(rectPreview.start.y, rectPreview.current.y);
	const start = gridToCanvas(minX, maxY, view, height);
	const end = gridToCanvas(maxX + 1, minY - 1, view, height);

	ctx.save();
	ctx.strokeStyle = renderTokens.selection;
	ctx.lineWidth = 2;
	ctx.setLineDash([6, 4]);
	ctx.strokeRect(start.x, start.y, end.x - start.x, end.y - start.y);
	ctx.restore();
}

export function drawFrame({ ctx, document, selectedObjectId, view, width, height, rectPreview, theme }: DrawFrameInput) {
	const themeTokens = getDesignThemeTokens(theme);
	const renderTokens = themeTokens.render;

	ctx.clearRect(0, 0, width, height);
	ctx.fillStyle = themeTokens.ui.canvas_bg;
	ctx.fillRect(0, 0, width, height);

	drawGrid(ctx, document, view, width, height, renderTokens);
	drawSurfaceEdges(ctx, document, view, height, renderTokens);
	if (document.metadata.mirrorEnabled) {
		drawSurfaceEdges(ctx, document, view, height, renderTokens, { mirrored: true, preview: true });
	}
	drawSelectionBox(ctx, document, selectedObjectId, view, height, renderTokens);
	drawRectPreview(ctx, view, height, rectPreview, renderTokens);
	drawMetadataMarkers(ctx, document, view, height, renderTokens);
}
