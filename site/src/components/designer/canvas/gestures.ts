import { CANVAS_PADDING } from '../constants';
import type { GridPoint, ViewState } from '../types';
import { MAX_VIEW_SCALE, MIN_VIEW_SCALE } from './draw';

// Preserve the world point under the gesture's centre as fingers move and scale.
export function transformView(view: ViewState, height: number, from: GridPoint, to: GridPoint, factor: number): ViewState {
	const scale = Math.min(MAX_VIEW_SCALE, Math.max(MIN_VIEW_SCALE, view.scale * factor));
	const ratio = scale / view.scale;
	return {
		scale,
		offsetX: to.x - CANVAS_PADDING - (from.x - CANVAS_PADDING - view.offsetX) * ratio,
		offsetY: to.y - height + CANVAS_PADDING + (height - CANVAS_PADDING + view.offsetY - from.y) * ratio,
	};
}

// Fill skipped cells during fast strokes, including on low-frequency touch devices.
export function strokeCells(from: GridPoint, to: GridPoint): GridPoint[] {
	const steps = Math.max(Math.abs(to.x - from.x), Math.abs(to.y - from.y));
	if (!steps) return [to];
	return Array.from({ length: steps }, (_, index) => ({
		x: Math.round(from.x + (to.x - from.x) * (index + 1) / steps),
		y: Math.round(from.y + (to.y - from.y) * (index + 1) / steps),
	}));
}
