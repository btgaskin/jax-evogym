import {
	ArrowClockwise,
	CopySimple,
	FlipHorizontal,
	Trash,
} from '@phosphor-icons/react';

import {
	SLOPE_SWATCH_POINTS,
	GROUP_HOTKEYS,
	VOXEL_DEFINITION_BY_TYPE,
	VOXEL_GROUPS,
} from './constants';
import { cn } from './lib/cn';
import { useEditorDispatch, useEditorState } from './state/context';
import type { GroupKey, VoxelType } from './types';

function VoxelSwatch({ type, size = 12 }: { type: VoxelType; size?: number }) {
	const definition = VOXEL_DEFINITION_BY_TYPE[type];
	const slopePoints = SLOPE_SWATCH_POINTS[type];

	if (slopePoints) {
		return (
			<svg width={size} height={size} viewBox="0 0 12 12" className="shrink-0">
				<polygon points={slopePoints} fill={definition?.hex ?? '#535B64'} />
			</svg>
		);
	}

	if (definition?.kind === 'gradient') {
		return (
			<span
				className="shrink-0 rounded-full border border-black/5"
				style={{
					width: size,
					height: size,
					background: `linear-gradient(180deg, ${definition.start} 0%, ${definition.end} 100%)`,
				}}
			/>
		);
	}

	return (
		<span
			className="shrink-0 rounded-full border border-black/5"
			style={{
				width: size,
				height: size,
				background: definition?.hex ?? '#c5cbc4',
			}}
		/>
	);
}

function TypeButton({ voxelType }: { voxelType: VoxelType }) {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const definition = VOXEL_DEFINITION_BY_TYPE[voxelType];
	const isActive = state.selectedVoxelType === voxelType;

	return (
		<button
			type="button"
			aria-pressed={isActive}
			title={`Draw with ${definition?.label}`}
			className={cn(
				'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium transition active:translate-y-px',
				isActive
					? 'border-accent bg-accent text-on-accent'
					: 'border-border bg-surface text-ink hover:border-accent hover:text-accent',
			)}
			onClick={() => dispatch({ type: 'SELECT_VOXEL_TYPE', voxelType })}
		>
			<VoxelSwatch type={voxelType} />
			<span>{definition?.label}</span>
		</button>
	);
}

function GroupCluster({
	groupKey,
	label,
	types,
	hotkey,
}: {
	groupKey: GroupKey;
	label: string;
	types: readonly VoxelType[];
	hotkey: string;
}) {
	return (
		<div className="flex shrink-0 items-center gap-2" role="group" aria-label={label}>
			<span className="text-xs font-semibold uppercase tracking-wide text-muted" title={`${label} (${hotkey})`}>
				{label}
			</span>
			<div className="flex items-center gap-1.5">
				{types.map((voxelType) => (
					<TypeButton key={voxelType} voxelType={voxelType} />
				))}
			</div>
		</div>
	);
}

function ObjectActions() {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const hasSelection = !!state.selectedObjectId;
	const objectId = state.selectedObjectId ?? '';

	return (
		<div
			className={cn(
				'flex items-center gap-1 transition-all duration-200',
				hasSelection
					? 'opacity-100'
					: 'pointer-events-none w-0 overflow-hidden opacity-0',
			)}
		>
			<div className="mr-2 h-5 w-px shrink-0 bg-border/60" />
			<button
				type="button"
				aria-label="Rotate selected object clockwise"
				title="Rotate clockwise (R)"
				className="inline-flex size-8 shrink-0 items-center justify-center rounded-full border border-border bg-surface text-muted transition hover:border-accent hover:text-accent active:translate-y-px"
				onClick={() => dispatch({ type: 'ROTATE_OBJECT', objectId })}
			>
				<ArrowClockwise size={14} />
			</button>
			<button
				type="button"
				aria-label="Mirror selected object horizontally"
				title="Mirror (Ctrl+M)"
				className="inline-flex size-8 shrink-0 items-center justify-center rounded-full border border-border bg-surface text-muted transition hover:border-accent hover:text-accent active:translate-y-px"
				onClick={() => dispatch({ type: 'MIRROR_OBJECT', objectId })}
			>
				<FlipHorizontal size={14} />
			</button>
			<button
				type="button"
				aria-label="Duplicate selected object"
				title="Duplicate (Ctrl+D)"
				className="inline-flex size-8 shrink-0 items-center justify-center rounded-full border border-border bg-surface text-muted transition hover:border-accent hover:text-accent active:translate-y-px"
				onClick={() => dispatch({ type: 'DUPLICATE_OBJECT', objectId })}
			>
				<CopySimple size={14} />
			</button>
			<button
				type="button"
				aria-label="Delete selected object"
				title="Delete (Del)"
				className="inline-flex size-8 shrink-0 items-center justify-center rounded-full border border-danger/25 bg-danger/6 text-danger transition hover:border-danger active:translate-y-px"
				onClick={() => dispatch({ type: 'DELETE_OBJECT', objectId })}
			>
				<Trash size={14} />
			</button>
		</div>
	);
}

export default function VoxelPalette() {
	return (
		<section className="designer-panel designer-scroll flex min-w-0 items-center gap-4 overflow-x-auto whitespace-nowrap rounded-[1.5rem] px-4 py-3">
			{VOXEL_GROUPS.map((group, index) => (
				<GroupCluster
					key={group.key}
					groupKey={group.key}
					label={group.label}
					types={group.types}
					hotkey={GROUP_HOTKEYS[index]?.toUpperCase() ?? String(index + 1)}
				/>
			))}

			<ObjectActions />
		</section>
	);
}
