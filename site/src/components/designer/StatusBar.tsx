import { BookOpenText, Question } from '@phosphor-icons/react';

import { SLOPE_SWATCH_POINTS, VOXEL_DEFINITION_BY_TYPE } from './constants';
import { useEditorState } from './state/context';
import type { CursorState, ViewState } from './types';

interface StatusBarProps {
	cursor: CursorState | null;
	selectedObjectName: string | null;
	view: ViewState;
	onToggleShortcuts: () => void;
}

export default function StatusBar({ cursor, selectedObjectName, view, onToggleShortcuts }: StatusBarProps) {
	const state = useEditorState();
	const activeType = state.selectedVoxelType;
	const typeDef = VOXEL_DEFINITION_BY_TYPE[activeType];
	const slopePoints = SLOPE_SWATCH_POINTS[activeType];
	const base = import.meta.env.BASE_URL.replace(/\/$/, '');

	return (
		<footer className="designer-scroll flex min-w-0 items-center gap-2 sm:gap-4 overflow-x-auto whitespace-nowrap border-t border-border/80 bg-panel/90 px-3 py-1 text-sm text-muted md:px-5">
			<span className="hidden sm:inline-flex items-center gap-1.5">
				{slopePoints ? (
					<svg width={10} height={10} viewBox="0 0 12 12" className="shrink-0">
						<polygon points={slopePoints} fill={typeDef?.hex ?? '#535B64'} />
					</svg>
				) : (
					<span
						className="inline-block size-2.5 shrink-0 rounded-full"
						style={{
							background:
								typeDef?.kind === 'gradient'
									? `linear-gradient(180deg, ${typeDef.start} 0%, ${typeDef.end} 100%)`
									: typeDef?.hex ?? '#c5cbc4',
						}}
					/>
				)}
				<span className="font-medium text-ink">{typeDef?.label}</span>
			</span>

			<span className="hidden sm:block h-3 w-px bg-border/60" />

			<span className="min-w-0 max-w-32 truncate" title={selectedObjectName ?? undefined}>
				{selectedObjectName ? (
					<>
						Sel: <span className="font-medium text-ink">{selectedObjectName}</span>
					</>
				) : (
					'No selection'
				)}
			</span>

			<span className="hidden sm:block h-3 w-px bg-border/60" />

			<span className="hidden sm:inline font-mono tabular-nums">
				{cursor ? `${cursor.x}, ${cursor.y}` : '-, -'}
			</span>

			<span className="hidden sm:block h-3 w-px bg-border/60" />

			<span className="font-mono tabular-nums">{Math.round(view.scale * 100)}%</span>

			<a
				href={`${base}/designer/designer-guide/`}
				className="ml-auto inline-flex min-h-11 shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-xs font-medium text-muted transition hover:bg-canvas hover:text-accent"
			>
				<BookOpenText size={14} weight="bold" />
				Guide
			</a>

			<button
				type="button"
				aria-label="Toggle keyboard shortcuts"
				title="Keyboard shortcuts (?)"
				className="inline-flex size-11 shrink-0 items-center justify-center rounded-full text-muted transition hover:text-accent"
				onClick={onToggleShortcuts}
			>
				<Question size={14} weight="bold" />
			</button>
		</footer>
	);
}
