import { useEffect, useState } from 'react';

import { Dialog } from '@base-ui-components/react/dialog';
import {
	ArrowClockwise,
	ArrowCounterClockwise,
	ArrowsOutSimple,
	CheckCircle,
	DownloadSimple,
	UploadSimple,
} from '@phosphor-icons/react';

import { cn } from './lib/cn';
import { MAX_GRID_DIM } from './constants';
import { useEditorDispatch, useEditorState } from './state/context';

interface TopBarProps {
	documentHandle: {
		fileName: string;
	} | null;
	newDocumentDialogOpen: boolean;
	onNewDocumentDialogOpenChange: (open: boolean) => void;
	onNewDocument: (width: number, height: number) => void;
	onOpenDocument: () => void | Promise<void>;
	onRequestFitView: () => void;
	onSaveDocument: () => void | Promise<void>;
	onValidate: () => void;
}

function ActionButton({
	children,
	className,
	...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
	return (
		<button
			type="button"
			className={cn(
				'inline-flex min-h-11 min-w-11 justify-center items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1.5 text-sm font-medium text-ink transition hover:border-accent hover:text-accent active:translate-y-px disabled:opacity-40 disabled:pointer-events-none',
				className,
			)}
			{...props}
		>
			{children}
		</button>
	);
}

export default function TopBar({
	documentHandle,
	newDocumentDialogOpen,
	onNewDocumentDialogOpenChange,
	onNewDocument,
	onOpenDocument,
	onRequestFitView,
	onSaveDocument,
	onValidate,
}: TopBarProps) {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const [width, setWidth] = useState(String(state.document.gridWidth));
	const [height, setHeight] = useState(String(state.document.gridHeight));

	useEffect(() => {
		if (!newDocumentDialogOpen) {
			setWidth(String(state.document.gridWidth));
			setHeight(String(state.document.gridHeight));
		}
	}, [newDocumentDialogOpen, state.document.gridHeight, state.document.gridWidth]);

	return (
		<div className="designer-scroll flex items-center gap-1 sm:gap-2 overflow-x-auto whitespace-nowrap border-b border-border/80 bg-panel/90 px-2 py-1 sm:py-2 md:px-5">
			<div className="flex items-center gap-1">
				<ActionButton
					aria-label="Undo (Ctrl+Z)"
					title="Undo (Ctrl+Z)"
					disabled={state.history.length === 0}
					onClick={() => dispatch({ type: 'UNDO' })}
				>
					<ArrowCounterClockwise size={16} />
				</ActionButton>
				<ActionButton
					aria-label="Redo (Ctrl+Shift+Z)"
					title="Redo (Ctrl+Shift+Z)"
					disabled={state.future.length === 0}
					onClick={() => dispatch({ type: 'REDO' })}
				>
					<ArrowClockwise size={16} />
				</ActionButton>
			</div>

			<div className="hidden sm:block h-5 w-px bg-border/60" />

			<div className="flex items-center gap-1">
				<Dialog.Root open={newDocumentDialogOpen} onOpenChange={onNewDocumentDialogOpenChange}>
					<Dialog.Trigger className="inline-flex min-h-11 items-center gap-1.5 rounded-full bg-accent px-3 py-1.5 text-sm font-medium text-on-accent transition hover:opacity-90 active:translate-y-px">
						New
					</Dialog.Trigger>
					<Dialog.Portal>
						<Dialog.Backdrop className="dialog-backdrop fixed inset-0 z-40 bg-ink/30" />
						<Dialog.Popup className="dialog-popup fixed left-1/2 top-1/2 z-50 w-[min(92vw,28rem)] -translate-x-1/2 -translate-y-1/2 rounded-[1.6rem] border border-border bg-panel p-6 shadow-2xl">
							<Dialog.Title className="text-xl font-semibold text-ink">
								Create a new document
							</Dialog.Title>
							<Dialog.Description className="mt-2 text-sm text-muted text-pretty">
								Choose the world bounds for the new design. This replaces the current draft.
							</Dialog.Description>
							<div className="mt-5 grid gap-4 sm:grid-cols-2">
								<label className="grid gap-2 text-sm font-medium text-ink">
									Grid width
									<input
										type="number"
										min={1}
										max={MAX_GRID_DIM}
										step={1}
										value={width}
										onChange={(event) => setWidth(event.target.value)}
										className="rounded-2xl border border-border bg-surface px-4 py-3 text-ink outline-none transition focus:border-accent focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1"
									/>
								</label>
								<label className="grid gap-2 text-sm font-medium text-ink">
									Grid height
									<input
										type="number"
										min={1}
										max={MAX_GRID_DIM}
										step={1}
										value={height}
										onChange={(event) => setHeight(event.target.value)}
										className="rounded-2xl border border-border bg-surface px-4 py-3 text-ink outline-none transition focus:border-accent focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1"
									/>
								</label>
							</div>
							<div className="mt-6 flex justify-end gap-2">
								<Dialog.Close className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-medium text-ink transition hover:border-accent hover:text-accent">
									Cancel
								</Dialog.Close>
								<button
									type="button"
									className="rounded-full bg-accent px-4 py-2 text-sm font-medium text-on-accent transition hover:opacity-90"
									onClick={() => {
										onNewDocument(
											Number(width) || state.document.gridWidth,
											Number(height) || state.document.gridHeight,
										);
										onNewDocumentDialogOpenChange(false);
									}}
								>
									Replace draft
								</button>
							</div>
						</Dialog.Popup>
					</Dialog.Portal>
				</Dialog.Root>
				<ActionButton aria-label="Import" title="Import" onClick={() => void onOpenDocument()}>
					<UploadSimple size={14} weight="bold" />
					<span className="hidden sm:inline">Import</span>
				</ActionButton>
				<ActionButton aria-label="Export" title="Export" onClick={() => void onSaveDocument()}>
					<DownloadSimple size={14} weight="bold" />
					<span className="hidden sm:inline">Export</span>
				</ActionButton>
			</div>

			<div className="hidden sm:block h-5 w-px bg-border/60" />

			<div className="flex items-center gap-1">
				<ActionButton className="hidden sm:inline-flex" onClick={onValidate}>
					<CheckCircle size={14} weight="bold" />
					Validate
				</ActionButton>
				<ActionButton aria-label="Fit" title="Fit canvas" onClick={onRequestFitView}>
					<ArrowsOutSimple size={14} weight="bold" />
					<span className="hidden sm:inline">Fit</span>
				</ActionButton>
			</div>
			{documentHandle ? (
				<>
					<div className="hidden sm:block h-5 w-px bg-border/60" />
					<span className="hidden sm:inline rounded-full border border-border/70 bg-surface px-3 py-1.5 text-xs font-medium text-muted">
						{documentHandle.fileName}
					</span>
				</>
			) : null}
		</div>
	);
}
