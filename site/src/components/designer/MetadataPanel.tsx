import { useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

import { ArrowsLeftRight, CaretDown, CaretRight, Compass, CrosshairSimple, X } from '@phosphor-icons/react';

import { cn } from './lib/cn';
import { useEditorDispatch, useEditorState } from './state/context';
import type { WorldDirection, WorldPoint } from './types';

type MarkerKey = 'spawn' | 'target';

interface MetadataPanelProps {
	markerPlacementMode: MarkerKey | null;
	setMarkerPlacementMode: Dispatch<SetStateAction<MarkerKey | null>>;
}

const DIRECTION_OPTIONS: WorldDirection[] = ['north', 'east', 'south', 'west'];

function MarkerCard({
	label,
	marker,
	point,
	markerPlacementMode,
	setMarkerPlacementMode,
	onSetPoint,
	onClearPoint,
}: {
	label: string;
	marker: MarkerKey;
	point: WorldPoint | null;
	markerPlacementMode: MarkerKey | null;
	setMarkerPlacementMode: Dispatch<SetStateAction<MarkerKey | null>>;
	onSetPoint: (marker: MarkerKey, point: WorldPoint) => void;
	onClearPoint: (marker: MarkerKey) => void;
}) {
	const isPlacing = markerPlacementMode === marker;

	return (
		<div className="rounded-[1.15rem] border border-border bg-surface p-3">
			<div className="flex items-center justify-between gap-2">
				<p className="text-xs font-semibold uppercase text-muted">{label}</p>
				{point ? (
					<span className="font-mono text-xs text-muted tabular-nums">
						{point.x}, {point.y}
					</span>
				) : (
					<span className="text-xs text-muted">Unset</span>
				)}
			</div>
			<div className="mt-3 grid gap-2 sm:grid-cols-[1fr_1fr]">
				<label className="grid min-w-0 gap-1 text-xs font-medium text-muted">
					X
					<input
						type="number"
						step={1}
						value={point?.x ?? ''}
						disabled={!point}
						onChange={(event) => {
							if (!point || event.target.value.trim() === '') {
								return;
							}
							const parsed = Number(event.target.value);
							if (!Number.isFinite(parsed)) {
								return;
							}
							onSetPoint(marker, { ...point, x: Math.floor(parsed) });
						}}
						className="w-full min-w-0 rounded-xl border border-border bg-panel px-3 py-2 text-sm text-ink outline-none transition focus:border-accent focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
					/>
				</label>
				<label className="grid min-w-0 gap-1 text-xs font-medium text-muted">
					Y
					<input
						type="number"
						step={1}
						value={point?.y ?? ''}
						disabled={!point}
						onChange={(event) => {
							if (!point || event.target.value.trim() === '') {
								return;
							}
							const parsed = Number(event.target.value);
							if (!Number.isFinite(parsed)) {
								return;
							}
							onSetPoint(marker, { ...point, y: Math.floor(parsed) });
						}}
						className="w-full min-w-0 rounded-xl border border-border bg-panel px-3 py-2 text-sm text-ink outline-none transition focus:border-accent focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
					/>
				</label>
			</div>
			<div className="mt-3 flex flex-wrap gap-2">
				<button
					type="button"
					className={cn(
						'inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition',
						isPlacing
							? 'border-accent bg-accent text-on-accent'
							: 'border-border bg-panel text-ink hover:border-accent hover:text-accent',
					)}
					onClick={() => setMarkerPlacementMode((current) => (current === marker ? null : marker))}
				>
					<CrosshairSimple size={14} />
					{isPlacing ? 'Cancel placement' : 'Place on canvas'}
				</button>
				<button
					type="button"
					className="inline-flex items-center gap-1.5 rounded-full border border-border bg-panel px-3 py-1.5 text-xs font-medium text-ink transition hover:border-accent hover:text-accent disabled:pointer-events-none disabled:opacity-40"
					disabled={!point}
					onClick={() => onClearPoint(marker)}
				>
					<X size={14} />
					Clear
				</button>
			</div>
		</div>
	);
}

export default function MetadataPanel({ markerPlacementMode, setMarkerPlacementMode }: MetadataPanelProps) {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const [collapsed, setCollapsed] = useState(true);
	const { direction, mirrorEnabled, spawn, target } = state.document.metadata;

	const setPoint = (marker: MarkerKey, point: WorldPoint) => {
		if (marker === 'spawn') {
			dispatch({ type: 'SET_SPAWN_POINT', point });
			return;
		}
		dispatch({ type: 'SET_TARGET_POINT', point });
	};

	const clearPoint = (marker: MarkerKey) => {
		if (marker === 'spawn') {
			dispatch({ type: 'CLEAR_SPAWN_POINT' });
			return;
		}
		dispatch({ type: 'CLEAR_TARGET_POINT' });
	};

	const summaryParts: string[] = [];
	if (spawn) summaryParts.push(`spawn ${spawn.x},${spawn.y}`);
	if (target) summaryParts.push(`target ${target.x},${target.y}`);
	if (direction) summaryParts.push(direction.slice(0, 1).toUpperCase());
	const summary = summaryParts.length > 0 ? summaryParts.join(' · ') : 'Unset';

	return (
		<section className="designer-panel w-full min-w-0 overflow-x-hidden rounded-[1.5rem] p-4">
			<button
				type="button"
				className="flex w-full items-center gap-2 text-left"
				aria-expanded={!collapsed}
				onClick={() => setCollapsed((value) => !value)}
			>
				{collapsed ? <CaretRight size={14} className="text-muted" /> : <CaretDown size={14} className="text-muted" />}
				<h2 className="text-sm font-semibold uppercase text-muted">Metadata</h2>
				<span className="ml-auto truncate font-mono text-xs text-muted">{summary}</span>
			</button>

			{collapsed ? null : (
			<>
			<p className="mt-2 text-xs text-muted text-pretty">
				Design annotations only. Mark a proposed spawn, target, or direction. These markers do not move objects or configure Python simulations.
			</p>

			<div className="mt-3 grid gap-2">
				<MarkerCard
					label="Spawn"
					marker="spawn"
					point={spawn}
					markerPlacementMode={markerPlacementMode}
					setMarkerPlacementMode={setMarkerPlacementMode}
					onSetPoint={setPoint}
					onClearPoint={clearPoint}
				/>
				<MarkerCard
					label="Target"
					marker="target"
					point={target}
					markerPlacementMode={markerPlacementMode}
					setMarkerPlacementMode={setMarkerPlacementMode}
					onSetPoint={setPoint}
					onClearPoint={clearPoint}
				/>
			</div>

			<div className="mt-2 rounded-[1.15rem] border border-border bg-surface p-3">
				<p className="text-xs font-semibold uppercase text-muted">Direction</p>
				<div className="mt-2 grid grid-cols-4 gap-1.5">
					{DIRECTION_OPTIONS.map((option) => (
						<button
							key={option}
							type="button"
							aria-label={`Set direction ${option}`}
							className={cn(
								'rounded-xl border px-2 py-2 text-xs font-medium uppercase transition',
								direction === option
									? 'border-accent bg-accent text-on-accent'
									: 'border-border bg-panel text-ink hover:border-accent hover:text-accent',
							)}
							onClick={() => dispatch({ type: 'SET_DIRECTION', direction: option })}
						>
							{option.slice(0, 1)}
						</button>
					))}
				</div>
				<button
					type="button"
					className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-border bg-panel px-3 py-1.5 text-xs font-medium text-ink transition hover:border-accent hover:text-accent"
					onClick={() => dispatch({ type: 'SET_DIRECTION', direction: null })}
				>
					<Compass size={14} />
					Clear direction
				</button>
			</div>

			<div className="mt-2 flex items-center justify-between gap-3 rounded-[1.15rem] border border-border bg-surface p-3">
				<div className="min-w-0">
					<p className="text-xs font-semibold uppercase text-muted">Mirror Preview</p>
					<p className="mt-1 text-xs text-muted text-pretty">Mirrors markers and heading on canvas.</p>
				</div>
				<label className="inline-flex items-center gap-2 text-xs font-medium text-ink">
					<ArrowsLeftRight size={14} />
					<input
						type="checkbox"
						checked={mirrorEnabled}
						onChange={(event) => dispatch({ type: 'SET_MIRROR_ENABLED', enabled: event.target.checked })}
						className="size-4 rounded border-border text-accent focus:ring-accent"
					/>
				</label>
			</div>

			{markerPlacementMode && (
				<div className="mt-3 rounded-xl border border-accent/30 bg-canvas px-3 py-2 text-xs text-ink">
					Click a grid cell on the canvas to set the {markerPlacementMode} marker.
				</div>
			)}
			</>
			)}
		</section>
	);
}
