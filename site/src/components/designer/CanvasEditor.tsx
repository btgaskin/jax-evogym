import { useEffect, useRef, useState } from 'react';
import type {
	Dispatch,
	KeyboardEvent as ReactKeyboardEvent,
	PointerEvent,
	SetStateAction,
	WheelEvent,
} from 'react';

import { Cube, Minus, Plus } from '@phosphor-icons/react';

import type { ThemeMode } from './constants';
import { MAX_VIEW_SCALE, MIN_VIEW_SCALE, canvasToGrid, drawFrame, getFittedView } from './canvas/draw';
import { useCanvasDPI } from './canvas/useCanvasDPI';
import { getBrowserThemeMode, subscribeBrowserTheme } from './host';
import { findObjectAt } from './lib/document';
import { transformView, strokeCells } from './canvas/gestures';
import { useEditorDispatch, useEditorState } from './state/context';
import type { CursorState, GridPoint, RectPreview, ViewState } from './types';

export type EditingTool = 'draw' | 'erase' | 'move';

type MarkerKey = 'spawn' | 'target';

type DragState =
	| { type: 'pan'; startX: number; startY: number; originX: number; originY: number }
	| { type: 'move'; objectId: string; lastCell: GridPoint }
	| { type: 'move-marker'; marker: MarkerKey; lastCell: GridPoint }
	| { type: 'rect'; start: GridPoint; current: GridPoint }
	| { type: 'paint' | 'erase' }
	| null;

interface CanvasEditorProps {
	tool: EditingTool;
	cursor: CursorState | null;
	fitSignal: number;
	markerPlacementMode: MarkerKey | null;
	setCursor: Dispatch<SetStateAction<CursorState | null>>;
	setMarkerPlacementMode: Dispatch<SetStateAction<MarkerKey | null>>;
	setView: Dispatch<SetStateAction<ViewState>>;
	view: ViewState;
}

export default function CanvasEditor({
	tool,
	cursor,
	fitSignal,
	markerPlacementMode,
	setCursor,
	setMarkerPlacementMode,
	setView,
	view,
}: CanvasEditorProps) {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const containerRef = useRef<HTMLDivElement | null>(null);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const dragRef = useRef<DragState>(null);
	const touches = useRef(new Map<number, { x: number; y: number }>());
	const pendingTouch = useRef<PointerEvent<HTMLCanvasElement> | null>(null);
	const pinching = useRef(false);
	const pinch = useRef<{ center: GridPoint; distance: number; view: ViewState } | null>(null);
	const fitStateRef = useRef({ initialized: false, signal: 0 });
	const spaceHeldRef = useRef(false);
	const lastPaintCellRef = useRef<GridPoint | null>(null);
	const [rectPreview, setRectPreview] = useState<RectPreview | null>(null);
	const [cursorStyle, setCursorStyle] = useState('crosshair');
	const [theme, setTheme] = useState<ThemeMode>(() => getBrowserThemeMode());
	const metrics = useCanvasDPI(containerRef, canvasRef);

	const setMarkerPoint = (marker: MarkerKey, point: GridPoint) => {
		if (marker === 'spawn') {
			dispatch({ type: 'SET_SPAWN_POINT', point });
			return;
		}
		dispatch({ type: 'SET_TARGET_POINT', point });
	};

	const markerAtPoint = (grid: GridPoint): MarkerKey | null => {
		const { spawn, target } = state.document.metadata;
		if (spawn && spawn.x === grid.x && spawn.y === grid.y) {
			return 'spawn';
		}
		if (target && target.x === grid.x && target.y === grid.y) {
			return 'target';
		}
		return null;
	};

	// Track Space key for paint-drag
	useEffect(() => {
		const onKeyDown = (event: KeyboardEvent) => {
			if (event.key === ' ') {
				spaceHeldRef.current = true;
			}
		};
		const onKeyUp = (event: KeyboardEvent) => {
			if (event.key === ' ') {
				spaceHeldRef.current = false;
				if (dragRef.current?.type === 'paint') {
					dragRef.current = null;
					lastPaintCellRef.current = null;
				}
			}
		};
		window.addEventListener('keydown', onKeyDown);
		window.addEventListener('keyup', onKeyUp);
		return () => {
			window.removeEventListener('keydown', onKeyDown);
			window.removeEventListener('keyup', onKeyUp);
		};
	}, []);

	useEffect(() => {
		if (!metrics.width || !metrics.height) {
			return;
		}

		if (!fitStateRef.current.initialized || fitStateRef.current.signal !== fitSignal) {
			fitStateRef.current = {
				initialized: true,
				signal: fitSignal,
			};
			setView(getFittedView(state.document, metrics.width, metrics.height));
		}
	}, [fitSignal, metrics.height, metrics.width, setView, state.document]);

	useEffect(() => {
		return subscribeBrowserTheme(setTheme);
	}, []);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas || !metrics.width || !metrics.height) {
			return;
		}

		const context = canvas.getContext('2d');
		if (!context) {
			return;
		}

		const frame = window.requestAnimationFrame(() => {
			drawFrame({
				ctx: context,
				document: state.document,
				selectedObjectId: state.selectedObjectId,
				view,
				width: metrics.width,
				height: metrics.height,
				rectPreview,
				theme,
			});
		});

		return () => window.cancelAnimationFrame(frame);
	}, [metrics.height, metrics.width, rectPreview, state.document, state.selectedObjectId, theme, view]);

	const getGrid = (event: PointerEvent<HTMLCanvasElement>) => {
		if (!canvasRef.current) {
			return null;
		}

		const grid = canvasToGrid(
			event.clientX,
			event.clientY,
			canvasRef.current.getBoundingClientRect(),
			view,
			metrics.height,
		);

		if (
			grid.x < 0 ||
			grid.y < 0 ||
			grid.x >= state.document.gridWidth ||
			grid.y >= state.document.gridHeight
		) {
			setCursor(null);
			return null;
		}

		setCursor((current) => current?.x === grid.x && current?.y === grid.y ? current : grid);
		return grid;
	};

	const updateCursorStyle = (grid: GridPoint | null) => {
		if (markerPlacementMode || spaceHeldRef.current || tool !== 'move') {
			setCursorStyle(tool === 'erase' ? 'cell' : 'crosshair');
			return;
		}
		if (!grid) {
			setCursorStyle('default');
			return;
		}
		if (markerAtPoint(grid)) {
			setCursorStyle('grab');
			return;
		}
		const objectUnder = findObjectAt(state.document, grid.x, grid.y, { visibleOnly: true });
		if (objectUnder) {
			if (objectUnder.id === state.selectedObjectId && !objectUnder.locked) {
				setCursorStyle('grab');
			} else {
				setCursorStyle('pointer');
			}
		} else {
			setCursorStyle('crosshair');
		}
	};

	const startEdit = (event: PointerEvent<HTMLCanvasElement>) => {
		const grid = getGrid(event);

		// Middle mouse button -> pan
		if (event.button === 1) {
			dragRef.current = {
				type: 'pan',
				startX: event.clientX,
				startY: event.clientY,
				originX: view.offsetX,
				originY: view.offsetY,
			};
			setCursorStyle('grabbing');
			return;
		}

		// Right-click -> erase
		if (event.button === 2) {
			if (grid) {
				dispatch({ type: 'ERASE_AT', worldX: grid.x, worldY: grid.y });
			}
			return;
		}
		if (event.button !== 0) {
			return;
		}

		if (markerPlacementMode) {
			if (grid) {
				setMarkerPoint(markerPlacementMode, grid);
				setMarkerPlacementMode(null);
			}
			return;
		}

		if (grid && tool === 'move') {
			const marker = markerAtPoint(grid);
			if (marker) {
				dragRef.current = { type: 'move-marker', marker, lastCell: grid };
				setCursorStyle('grabbing');
				return;
			}
		}

		// Keep the desktop rectangle shortcut in every tool.
		if (event.shiftKey && grid) {
			dragRef.current = { type: 'rect', start: grid, current: grid };
			setRectPreview({ start: grid, current: grid });
			return;
		}
		if (grid && (spaceHeldRef.current || tool !== 'move')) {
			const erase = tool === 'erase' && !spaceHeldRef.current;
			dragRef.current = { type: erase ? 'erase' : 'paint' };
			lastPaintCellRef.current = grid;
			dispatch({ type: erase ? 'ERASE_AT' : 'PAINT_AT', worldX: grid.x, worldY: grid.y });
			return;
		}

		if (!grid) {
			// Click outside grid -> deselect
			dispatch({ type: 'SELECT_OBJECT', objectId: null });
			// Start pan from outside grid
			dragRef.current = {
				type: 'pan',
				startX: event.clientX,
				startY: event.clientY,
				originX: view.offsetX,
				originY: view.offsetY,
			};
			return;
		}

		// Context-dependent click
		const objectUnder = findObjectAt(state.document, grid.x, grid.y, { visibleOnly: true });

		if (objectUnder) {
			// Click on object -> select it, maybe start move
			dispatch({ type: 'SELECT_OBJECT', objectId: objectUnder.id });
			if (!objectUnder.locked) {
				dragRef.current = { type: 'move', objectId: objectUnder.id, lastCell: grid };
				setCursorStyle('grabbing');
			}
		} else {
			// Move mode pans from empty space without changing the document.
			dispatch({ type: 'SELECT_OBJECT', objectId: null });
			dragRef.current = { type: 'pan', startX: event.clientX, startY: event.clientY, originX: view.offsetX, originY: view.offsetY };
		}
	};

	const touchGeometry = () => {
		const [a, b] = [...touches.current.values()];
		const rect = canvasRef.current!.getBoundingClientRect();
		return { center: { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top }, distance: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)) };
	};

	const onPointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
		event.currentTarget.setPointerCapture(event.pointerId);
		if (event.pointerType !== 'touch') { startEdit(event); return; }
		touches.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
		if (touches.current.size === 1 && !pinching.current) {
			// Defer a tap until release, so the first finger of a pinch cannot paint.
			pendingTouch.current = event;
		} else if (touches.current.size === 2) {
			pendingTouch.current = null;
			dragRef.current = null;
			setRectPreview(null);
			pinching.current = true;
			pinch.current = { ...touchGeometry(), view };
		}
	};

	const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
		if (event.pointerType === 'touch' && touches.current.has(event.pointerId)) {
			touches.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
			if (pinching.current) {
				if (touches.current.size === 2 && pinch.current) {
					const next = touchGeometry();
					const initial = pinch.current;
					setView(transformView(initial.view, metrics.height, initial.center, next.center, next.distance / initial.distance));
				}
				return;
			}
			const pending = pendingTouch.current;
			if (pending) {
				if (Math.hypot(event.clientX - pending.clientX, event.clientY - pending.clientY) < 6) return;
				pendingTouch.current = null;
				startEdit(pending);
			}
		}
		const grid = getGrid(event);
		const drag = dragRef.current;

		if (!drag) {
			updateCursorStyle(grid);
			return;
		}

		if (drag.type === 'pan') {
			setView({
				...view,
				offsetX: drag.originX + (event.clientX - drag.startX),
				offsetY: drag.originY + (event.clientY - drag.startY),
			});
			return;
		}

		if (!grid) {
			return;
		}

		if (drag.type === 'paint' || drag.type === 'erase') {
			const last = lastPaintCellRef.current;
			if (!last || grid.x !== last.x || grid.y !== last.y) {
				for (const cell of strokeCells(last ?? grid, grid)) {
					dispatch({ type: drag.type === 'erase' ? 'ERASE_AT' : 'PAINT_AT', worldX: cell.x, worldY: cell.y });
				}
				lastPaintCellRef.current = grid;
			}
			return;
		}
		if (drag.type === 'move-marker') {
			if (grid.x === drag.lastCell.x && grid.y === drag.lastCell.y) {
				return;
			}
			setMarkerPoint(drag.marker, grid);
			dragRef.current = { ...drag, lastCell: grid };
			return;
		}

		if (drag.type === 'move') {
			if (grid.x === drag.lastCell.x && grid.y === drag.lastCell.y) {
				return;
			}

			dispatch({
				type: 'MOVE_OBJECT_BY',
				objectId: drag.objectId,
				deltaX: grid.x - drag.lastCell.x,
				deltaY: grid.y - drag.lastCell.y,
			});

			dragRef.current = { ...drag, lastCell: grid };
			return;
		}

		if (drag.type === 'rect') {
			dragRef.current = { ...drag, current: grid };
			setRectPreview({ start: drag.start, current: grid });
		}
	};

	const clearDrag = (event: PointerEvent<HTMLCanvasElement>) => {
		if (pendingTouch.current && event.type === 'pointerup' && !pinching.current) startEdit(event);
		pendingTouch.current = null;
		touches.current.delete(event.pointerId);
		if (touches.current.size < 2) pinch.current = null;
		if (touches.current.size === 0) pinching.current = false;
		const drag = dragRef.current;
		if (drag?.type === 'rect' && event.type === 'pointerup') {
			dispatch({
				type: 'PAINT_RECT',
				start: drag.start,
				end: drag.current,
			});
		}

		dragRef.current = null;
		lastPaintCellRef.current = null;
		setRectPreview(null);
		setCursorStyle('crosshair');

		if (event.currentTarget.hasPointerCapture(event.pointerId)) {
			event.currentTarget.releasePointerCapture(event.pointerId);
		}
	};

	const onContextMenu = (event: React.MouseEvent<HTMLCanvasElement>) => {
		event.preventDefault();
	};

	const onCanvasKeyDown = (event: ReactKeyboardEvent<HTMLCanvasElement>) => {
		const key = event.key.toLowerCase();
		if (
			key === 'arrowleft' ||
			key === 'arrowright' ||
			key === 'arrowup' ||
			key === 'arrowdown'
		) {
			event.preventDefault();
			event.stopPropagation();
			const current = cursor ?? { x: 0, y: 0 };
			setCursor({
				x: Math.min(
					state.document.gridWidth - 1,
					Math.max(0, current.x + (key === 'arrowleft' ? -1 : key === 'arrowright' ? 1 : 0)),
				),
				y: Math.min(
					state.document.gridHeight - 1,
					Math.max(0, current.y + (key === 'arrowdown' ? -1 : key === 'arrowup' ? 1 : 0)),
				),
			});
			return;
		}

		if (event.key === ' ') {
			event.preventDefault();
			event.stopPropagation();
			if (cursor) {
				const occupant = findObjectAt(state.document, cursor.x, cursor.y, {
					visibleOnly: false,
				});
				if (!occupant || occupant.id === state.selectedObjectId) {
					dispatch({ type: 'PAINT_AT', worldX: cursor.x, worldY: cursor.y });
				}
			}
			return;
		}

		if (key === 'x') {
			event.preventDefault();
			event.stopPropagation();
			if (cursor) {
				dispatch({ type: 'ERASE_AT', worldX: cursor.x, worldY: cursor.y });
			}
		}
	};

	const onWheel = (event: WheelEvent<HTMLCanvasElement>) => {
		event.preventDefault();
		if (!canvasRef.current) {
			return;
		}

		const rect = canvasRef.current.getBoundingClientRect();
		const pointerX = event.clientX - rect.left;
		const pointerY = event.clientY - rect.top;
		setView(transformView(view, metrics.height, { x: pointerX, y: pointerY }, { x: pointerX, y: pointerY }, event.deltaY > 0 ? 0.92 : 1.08));
	};

	const hasObjects = state.document.objects.length > 0;

	return (
		<div className="designer-canvas designer-panel grid min-h-0 min-w-0 overflow-hidden rounded-[1.8rem] border border-border/80 bg-canvas shadow-sm">
			<div ref={containerRef} className="designer-grid-bg relative min-h-0 min-w-0">
				<canvas
					ref={canvasRef}
					aria-label="Voxel world canvas. Choose Draw, Erase, or Move; use two fingers to pan and zoom. Use arrow keys to move the grid cursor, Space to place a voxel, and X to erase."
					className="block size-full focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
					role="application"
					style={{ cursor: cursorStyle, touchAction: 'none' }}
					tabIndex={0}
					onContextMenu={onContextMenu}
					onFocus={() => setCursor((current) => current ?? { x: 0, y: 0 })}
					onKeyDown={onCanvasKeyDown}
					onPointerCancel={clearDrag}
					onLostPointerCapture={clearDrag}
					onPointerDown={onPointerDown}
					onPointerMove={onPointerMove}
					onPointerUp={clearDrag}
					onWheel={onWheel}
				/>
				<div className="absolute right-2 top-2 flex rounded-full border border-border bg-panel shadow-sm" role="group" aria-label="Canvas zoom">
					{[{ label: 'Zoom out', factor: 0.8, Icon: Minus }, { label: 'Zoom in', factor: 1.25, Icon: Plus }].map(({ label, factor, Icon }) => (
						<button key={label} type="button" aria-label={label} title={label} className="inline-flex size-11 items-center justify-center rounded-full text-ink hover:bg-surface disabled:opacity-40"
							disabled={factor > 1 ? view.scale >= MAX_VIEW_SCALE : view.scale <= MIN_VIEW_SCALE}
							onClick={() => { const center = { x: metrics.width / 2, y: metrics.height / 2 }; setView(current => transformView(current, metrics.height, center, center, factor)); }}><Icon size={18} /></button>
					))}
				</div>
				{markerPlacementMode && (
					<div className="absolute bottom-2 left-2 right-2 flex items-center justify-between gap-2 rounded-2xl border border-border bg-panel px-3 text-sm text-ink" role="status">
						<span>Tap a grid cell to place {markerPlacementMode}.</span>
						<button type="button" className="min-h-11 px-2 font-medium text-accent" onClick={() => setMarkerPlacementMode(null)}>Cancel</button>
					</div>
				)}
				{!hasObjects && !markerPlacementMode && (
					<div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 text-center">
						<div className="rounded-[1.25rem] border border-dashed border-border/50 bg-panel/80 px-4 py-3 sm:px-10 sm:py-8">
							<Cube size={48} weight="thin" className="mx-auto text-muted/30" />
							<p className="mt-3 text-sm font-medium text-ink">Draw your first voxel</p>
							<p className="mt-2 max-w-xl text-xs text-muted">
								Build robots and terrain from voxels, then export world JSON for the jax-evogym simulator.
							</p>
							<p className="mt-2 max-w-xl text-xs text-muted">
								Choose Draw and a material, then tap or drag on the grid. Pinch with two fingers to zoom and pan.
							</p>
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
