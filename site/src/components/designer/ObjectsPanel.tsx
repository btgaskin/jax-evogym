import { useEffect, useState } from 'react';

import { AlertDialog } from '@base-ui-components/react/alert-dialog';
import {
	CursorClick,
	Eye,
	EyeSlash,
	Lock,
	LockOpen,
	PlusCircle,
	Trash,
} from '@phosphor-icons/react';

import { cn } from './lib/cn';
import { useEditorDispatch, useEditorState } from './state/context';
import type { EditorObject } from './types';

function ObjectRow({
	object,
	onDeleteDialogOpenChange,
}: {
	object: EditorObject;
	onDeleteDialogOpenChange: (open: boolean) => void;
}) {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const [name, setName] = useState(object.name);
	const [nameError, setNameError] = useState<string | null>(null);

	useEffect(() => {
		setName(object.name);
		setNameError(null);
	}, [object.name]);

	const voxelCount = object.voxels.length;
	const nameErrorId = `object-name-error-${object.id}`;
	const commitName = () => {
		if (!name.trim()) {
			setName(object.name);
			setNameError('Object name cannot be empty.');
			return;
		}
		setName(name.trim());
		setNameError(null);
		dispatch({ type: 'RENAME_OBJECT', objectId: object.id, name });
	};

	return (
		<div
			className={cn(
				'rounded-[1.25rem] border p-3 transition',
				state.selectedObjectId === object.id
					? 'border-accent bg-canvas'
					: 'border-border bg-surface',
			)}
		>
			<div className="flex items-start justify-between gap-3">
				<div className="min-w-0 flex-1">
					<input
						value={name}
						aria-describedby={nameError ? nameErrorId : undefined}
						aria-invalid={nameError ? true : undefined}
						onBlur={commitName}
						onChange={(event) => {
							setName(event.target.value);
							if (event.target.value.trim()) {
								setNameError(null);
							}
						}}
						className="w-full truncate rounded-xl border border-transparent bg-transparent px-2 py-1 text-sm font-medium text-ink outline-none transition focus:border-accent focus:bg-surface focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1"
					/>
					{nameError ? (
						<p id={nameErrorId} className="mt-1 px-2 text-xs text-danger">
							{nameError}
						</p>
					) : null}
					<p className="mt-1 text-xs text-muted">
						Origin <span className="font-mono tabular-nums">{object.origin.x}</span>,{' '}
						<span className="font-mono tabular-nums">{object.origin.y}</span>
					</p>
				</div>
				<span className="rounded-full bg-panel px-2.5 py-1 text-xs font-medium text-muted">
					{voxelCount} voxels
				</span>
			</div>
			<div className="mt-3 flex flex-wrap gap-2">
				<button
					type="button"
					className="inline-flex items-center gap-2 rounded-full border border-border bg-panel px-3 py-2 text-xs font-medium text-ink transition hover:border-accent hover:text-accent"
					onClick={() => dispatch({ type: 'SELECT_OBJECT', objectId: object.id })}
				>
					<CursorClick size={14} />
					Select
				</button>
				<button
					type="button"
					aria-label={object.visible ? 'Hide object' : 'Show object'}
					className="inline-flex size-9 items-center justify-center rounded-full border border-border bg-panel text-muted transition hover:border-accent hover:text-accent"
					onClick={() => dispatch({ type: 'TOGGLE_VISIBLE', objectId: object.id })}
				>
					{object.visible ? <Eye size={16} /> : <EyeSlash size={16} />}
				</button>
				<button
					type="button"
					aria-label={object.locked ? 'Unlock object' : 'Lock object'}
					className="inline-flex size-9 items-center justify-center rounded-full border border-border bg-panel text-muted transition hover:border-accent hover:text-accent"
					onClick={() => dispatch({ type: 'TOGGLE_LOCKED', objectId: object.id })}
				>
					{object.locked ? <Lock size={16} /> : <LockOpen size={16} />}
				</button>
				<AlertDialog.Root onOpenChange={onDeleteDialogOpenChange}>
					<AlertDialog.Trigger
						aria-label="Delete object"
						className="inline-flex size-9 items-center justify-center rounded-full border border-danger/25 bg-danger/6 text-danger transition hover:border-danger"
					>
						<Trash size={16} />
					</AlertDialog.Trigger>
					<AlertDialog.Portal>
						<AlertDialog.Backdrop className="dialog-backdrop fixed inset-0 z-40 bg-ink/30" />
						<AlertDialog.Popup className="dialog-popup fixed left-1/2 top-1/2 z-50 w-[min(92vw,28rem)] -translate-x-1/2 -translate-y-1/2 rounded-[1.6rem] border border-border bg-panel p-6 shadow-2xl">
							<AlertDialog.Title className="text-lg font-semibold text-ink">
								Delete {object.name}?
							</AlertDialog.Title>
							<AlertDialog.Description className="mt-2 text-sm text-muted text-pretty">
								This removes the object from the current draft. Undo is still available after the dialog closes.
							</AlertDialog.Description>
							<div className="mt-6 flex justify-end gap-2">
								<AlertDialog.Close className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-medium text-ink transition hover:border-accent hover:text-accent">
									Cancel
								</AlertDialog.Close>
								<AlertDialog.Close
									className="rounded-full bg-danger px-4 py-2 text-sm font-medium text-on-danger transition hover:opacity-90"
									onClick={() => dispatch({ type: 'DELETE_OBJECT', objectId: object.id })}
								>
									Delete object
								</AlertDialog.Close>
							</div>
						</AlertDialog.Popup>
					</AlertDialog.Portal>
				</AlertDialog.Root>
			</div>
		</div>
	);
}

export default function ObjectsPanel({
	onDeleteDialogOpenChange,
}: {
	onDeleteDialogOpenChange: (open: boolean) => void;
}) {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const count = state.document.objects.length;

	return (
		<section className="designer-panel w-full min-w-0 overflow-x-hidden rounded-[1.5rem] p-4">
			<div className="flex items-start justify-between gap-3">
				<h2 className="text-sm font-semibold uppercase text-muted">
					Objects{' '}
					<span className="ml-1 font-mono text-xs font-normal">({count})</span>
				</h2>
				<button
					type="button"
					className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1.5 text-xs font-medium text-ink transition hover:border-accent hover:text-accent"
					onClick={() => dispatch({ type: 'CREATE_OBJECT_AT', worldX: 0, worldY: 0 })}
				>
					<PlusCircle size={14} />
					Add
				</button>
			</div>
			<div className="mt-4 grid gap-3">
				{count ? (
					state.document.objects.map((object) => (
						<ObjectRow
							key={object.id}
							object={object}
							onDeleteDialogOpenChange={onDeleteDialogOpenChange}
						/>
					))
				) : (
					<div className="rounded-[1.25rem] border border-dashed border-border bg-surface p-5 text-sm text-muted">
						<p className="text-pretty">No objects yet. Click on the canvas to place your first voxel.</p>
					</div>
				)}
			</div>
		</section>
	);
}
