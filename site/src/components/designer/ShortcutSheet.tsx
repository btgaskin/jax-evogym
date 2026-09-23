import { Dialog } from '@base-ui-components/react/dialog';

const SHORTCUTS = [
	{ label: 'Select group', keys: 'Q / W (or 1 / 2)' },
	{ label: 'Cycle groups', keys: 'A / D (or Shift+ArrowUp / Shift+ArrowDown)' },
	{ label: 'Cycle type in group', keys: 'S' },
	{ label: 'Move selected object', keys: 'Arrow keys' },
	{ label: 'Place voxel', keys: 'Space / Click' },
	{ label: 'Paint stroke', keys: 'Space + Drag' },
	{ label: 'Erase voxel', keys: 'X / Right-click' },
	{ label: 'Rectangle fill', keys: 'Shift + Drag' },
	{ label: 'Pan viewport', keys: 'Drag outside grid / Middle-drag' },
	{ label: 'Zoom', keys: 'Scroll' },
	{ label: 'Rotate object', keys: 'R' },
	{ label: 'Mirror object', keys: 'Ctrl+M' },
	{ label: 'Duplicate object', keys: 'Ctrl+D' },
	{ label: 'Delete object', keys: 'Delete / Backspace' },
	{ label: 'Undo / Redo', keys: 'Ctrl+Z / Ctrl+Shift+Z or Ctrl+Y' },
	{ label: 'Show shortcuts', keys: '?' },
] as const;

interface ShortcutSheetProps {
	open: boolean;
	onOpenChange: (open: boolean) => void;
}

export default function ShortcutSheet({ open, onOpenChange }: ShortcutSheetProps) {
	return (
		<Dialog.Root open={open} onOpenChange={onOpenChange}>
			<Dialog.Portal>
				<Dialog.Backdrop className="dialog-backdrop fixed inset-0 z-40 bg-ink/30" />
				<Dialog.Popup className="dialog-popup fixed left-1/2 top-1/2 z-50 max-h-[90dvh] overflow-y-auto w-[min(92vw,36rem)] -translate-x-1/2 -translate-y-1/2 rounded-[1.6rem] border border-border bg-panel p-6 shadow-2xl">
					<Dialog.Title className="text-lg font-semibold text-ink">
						Keyboard shortcuts
					</Dialog.Title>
					<div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-2 text-sm">
						{SHORTCUTS.map((shortcut) => (
							<div
								key={shortcut.label}
								className="flex items-center justify-between gap-4 border-b border-border/40 py-1.5"
							>
								<span className="text-muted">{shortcut.label}</span>
								<kbd className="font-mono text-xs text-ink">{shortcut.keys}</kbd>
							</div>
						))}
					</div>
					<div className="mt-5 flex justify-end">
						<Dialog.Close className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-medium text-ink transition hover:border-accent hover:text-accent">
							Close
						</Dialog.Close>
					</div>
				</Dialog.Popup>
			</Dialog.Portal>
		</Dialog.Root>
	);
}
