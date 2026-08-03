import { CheckCircle, WarningCircle } from '@phosphor-icons/react';

import { cn } from './lib/cn';
import { useEditorDispatch, useEditorState } from './state/context';

const ISSUE_LABELS: Record<string, string> = {
	disconnected_object: 'Disconnected object',
	duplicate_name: 'Duplicate name',
	empty_object: 'Empty object',
	invalid_grid_size: 'Invalid grid size',
	no_actuator: 'No actuator',
	out_of_bounds: 'Voxel outside grid',
	overlap: 'Overlapping objects',
	slope2_invalid_pair: 'Invalid 2x1 slope',
	slope2_rotate_disabled: 'Rotation unavailable',
	slope3_invalid_triplet: 'Invalid 3x1 slope',
	slope_requires_static_object: 'Slope mixed with dynamic cells',
	spawn_out_of_bounds: 'Spawn outside grid',
	target_out_of_bounds: 'Target outside grid',
	unknown_voxel_type: 'Unknown voxel type',
};

function issueLabel(code: string) {
	return (
		ISSUE_LABELS[code] ??
		code
			.split('_')
			.map((part) => part.charAt(0).toUpperCase() + part.slice(1))
			.join(' ')
	);
}

export default function ValidationPanel() {
	const state = useEditorState();
	const dispatch = useEditorDispatch();

	const errorCount = state.issues.filter((issue) => issue.severity === 'error').length;
	const warningCount = state.issues.filter((issue) => issue.severity === 'warning').length;

	return (
		<section className="designer-panel w-full min-w-0 overflow-x-hidden rounded-[1.5rem] p-4">
			<div className="flex items-start justify-between gap-3">
				<h2 className="text-sm font-semibold uppercase text-muted">
					Validation
				</h2>
				{errorCount > 0 ? (
					<span className="rounded-full bg-danger/10 px-2.5 py-0.5 text-xs font-medium text-danger">
						{errorCount} error{errorCount !== 1 ? 's' : ''}
					</span>
				) : warningCount > 0 ? (
					<span className="rounded-full bg-amber-500/10 px-2.5 py-0.5 text-xs font-medium text-amber-600">
						{warningCount} warning{warningCount !== 1 ? 's' : ''}
					</span>
				) : (
					<span className="rounded-full bg-success/10 px-2.5 py-0.5 text-xs font-medium text-success">
						OK
					</span>
				)}
			</div>
			<div className="mt-4 grid gap-3">
				{state.issues.length ? (
					state.issues.map((issue) => (
						<button
							key={`${issue.code}-${issue.message}`}
							type="button"
							className={cn(
								'rounded-[1.25rem] border-l-4 bg-surface p-4 text-left transition hover:border-accent',
								issue.severity === 'error'
									? 'border-l-danger'
									: 'border-l-amber-500',
							)}
							onClick={() =>
								issue.objectId
									? dispatch({ type: 'SELECT_OBJECT', objectId: issue.objectId })
									: null
							}
						>
							<div className="flex items-center gap-2 text-sm font-semibold text-ink">
								<WarningCircle
									size={16}
									weight="fill"
									className={issue.severity === 'error' ? 'text-danger' : 'text-amber-500'}
								/>
							<span>{issueLabel(issue.code)}</span>
								<span className="rounded-full bg-panel px-2 py-0.5 text-[11px] uppercase tracking-[0.08em] text-muted">
									{issue.severity}
								</span>
							</div>
							<p className="mt-2 text-sm text-muted text-pretty">{issue.message}</p>
						</button>
					))
				) : (
					<div className="rounded-[1.25rem] border border-dashed border-success/25 bg-success/6 p-5 text-sm text-success">
						<div className="flex items-center gap-2 font-medium">
							<CheckCircle size={16} weight="fill" />
							No issues
						</div>
						<p className="mt-2 text-pretty text-success/80">Issues update automatically as you edit.</p>
					</div>
				)}
			</div>
		</section>
	);
}
