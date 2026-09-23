import { describe, expect, it } from 'vitest';
import { transformView, strokeCells } from '../components/designer/canvas/gestures';
import { gridPointToCanvas, MAX_VIEW_SCALE, MIN_VIEW_SCALE } from '../components/designer/canvas/draw';

describe('canvas gestures', () => {
	it('keeps the world point under a moving pinch centre', () => {
		const view = { scale: 0.7, offsetX: 30, offsetY: -20 };
		const from = gridPointToCanvas(6, 4, view, 500);
		const to = { x: from.x + 42, y: from.y - 17 };
		const next = transformView(view, 500, from, to, 1.8);
		const anchored = gridPointToCanvas(6, 4, next, 500);
		expect(anchored.x).toBeCloseTo(to.x);
		expect(anchored.y).toBeCloseTo(to.y);
	});
	it('pans without scaling and respects zoom limits without drifting', () => {
		const view = { scale: 1, offsetX: 0, offsetY: 0 };
		const point = gridPointToCanvas(3, 2, view, 300);
		expect(transformView(view, 300, point, { x: point.x + 10, y: point.y + 20 }, 1)).toEqual({ scale: 1, offsetX: 10, offsetY: 20 });
		for (const [factor, expected] of [[100, MAX_VIEW_SCALE], [0.001, MIN_VIEW_SCALE]]) {
			const next = transformView(view, 300, point, point, factor);
			expect(next.scale).toBe(expected);
			const anchored = gridPointToCanvas(3, 2, next, 300);
			expect(anchored.x).toBeCloseTo(point.x);
			expect(anchored.y).toBeCloseTo(point.y);
		}
	});
	it('fills skipped cells in horizontal and reverse diagonal strokes', () => {
		expect(strokeCells({x: 1, y: 2}, {x: 4, y: 2})).toEqual([{x: 2, y: 2}, {x: 3, y: 2}, {x: 4, y: 2}]);
		expect(strokeCells({x: 3, y: 3}, {x: 0, y: 0})).toEqual([{x: 2, y: 2}, {x: 1, y: 1}, {x: 0, y: 0}]);
		expect(strokeCells({x: 2, y: 2}, {x: 2, y: 2})).toEqual([{x: 2, y: 2}]);
	});
});
