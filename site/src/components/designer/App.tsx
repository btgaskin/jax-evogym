import { useEffect, useMemo, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

import { Dialog } from '@base-ui-components/react/dialog';
import { FadersHorizontal, X } from '@phosphor-icons/react';

import { getFittedView } from './canvas/draw';
import CanvasEditor from './CanvasEditor';
import DocumentPanel from './DocumentPanel';
import MetadataPanel from './MetadataPanel';
import ObjectsPanel from './ObjectsPanel';
import ShortcutSheet from './ShortcutSheet';
import StatusBar from './StatusBar';
import Toast from './Toast';
import TopBar from './TopBar';
import ValidationPanel from './ValidationPanel';
import VoxelPalette from './VoxelPalette';
import {
	browserStorageKey,
	exportBrowserDocument,
	openBrowserDocument,
	setBrowserPresentationState,
} from './host';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useAutoValidation } from './hooks/useAutoValidation';
import { createEmptyDocument } from './lib/document';
import { validateDocument } from './lib/validation';
import { EditorProvider, useEditor } from './state/context';
import type { CursorState, ViewState } from './types';

export interface SurfaceMessage {
	tone: 'error' | 'success';
	text: string;
}

function DesignerPanels({
	markerPlacementMode,
	onDeleteDialogOpenChange,
	setMarkerPlacementMode,
}: {
	markerPlacementMode: 'spawn' | 'target' | null;
	onDeleteDialogOpenChange: (open: boolean) => void;
	setMarkerPlacementMode: Dispatch<SetStateAction<'spawn' | 'target' | null>>;
}) {
	return (
		<>
			<DocumentPanel />
			<MetadataPanel
				markerPlacementMode={markerPlacementMode}
				setMarkerPlacementMode={setMarkerPlacementMode}
			/>
			<ObjectsPanel onDeleteDialogOpenChange={onDeleteDialogOpenChange} />
			<ValidationPanel />
		</>
	);
}

function DesignerShell() {
	const { state, dispatch } = useEditor();
	const [view, setView] = useState<ViewState>(() => getFittedView(state.document, 1100, 720));
	const [fitSignal, setFitSignal] = useState(0);
	const [cursor, setCursor] = useState<CursorState | null>(null);
	const [message, setMessage] = useState<SurfaceMessage | null>(null);
	const [showShortcuts, setShowShortcuts] = useState(false);
	const [markerPlacementMode, setMarkerPlacementMode] = useState<'spawn' | 'target' | null>(null);
	const [mobileControlsOpen, setMobileControlsOpen] = useState(false);
	const [newDocumentDialogOpen, setNewDocumentDialogOpen] = useState(false);
	const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
	const [documentHandle, setDocumentHandle] = useState<{
		fileName: string;
	} | null>(null);
	const baselineDocumentRef = useRef(JSON.stringify(state.document));
	const serializedDocument = useMemo(() => JSON.stringify(state.document), [state.document]);
	const dirty = serializedDocument !== baselineDocumentRef.current;

	const isModalOpen =
		showShortcuts || mobileControlsOpen || newDocumentDialogOpen || deleteDialogOpen;
	useKeyboardShortcuts(state, dispatch, cursor, setShowShortcuts, isModalOpen);
	useAutoValidation(state, dispatch);

	useEffect(() => {
		if (!message) {
			return;
		}

		const timeout = window.setTimeout(() => setMessage(null), 4200);
		return () => window.clearTimeout(timeout);
	}, [message]);

	useEffect(() => {
		if (!dirty) {
			return;
		}

		const onBeforeUnload = (event: BeforeUnloadEvent) => {
			event.preventDefault();
			event.returnValue = '';
		};

		window.addEventListener('beforeunload', onBeforeUnload);
		return () => window.removeEventListener('beforeunload', onBeforeUnload);
	}, [dirty]);

	useEffect(() => {
		setBrowserPresentationState({
			documentName: documentHandle?.fileName ?? null,
			dirty,
		});
	}, [dirty, documentHandle?.fileName]);

	const selectedObject = useMemo(
		() => state.document.objects.find((object) => object.id === state.selectedObjectId) ?? null,
		[state.document.objects, state.selectedObjectId],
	);

	const requestFitView = () => {
		setFitSignal((value) => value + 1);
	};

	const validateCurrentDocument = () => {
		const issues = validateDocument(state.document);
		dispatch({
			type: 'SET_ISSUES',
			issues,
		});
		return issues;
	};

	const onNewDocument = (width: number, height: number) => {
		const nextDocument = createEmptyDocument(width, height);
		dispatch({
			type: 'NEW_DOCUMENT',
			width,
			height,
		});
		dispatch({ type: 'SET_ISSUES', issues: [] });
		setDocumentHandle(null);
		baselineDocumentRef.current = JSON.stringify(nextDocument);
		setMessage(null);
		requestFitView();
	};

	const onOpenDocument = async () => {
		try {
			const result = await openBrowserDocument();
			if (!result) {
				return;
			}

			dispatch({ type: 'IMPORT_DOCUMENT', document: result.document });
			dispatch({
				type: 'SET_ISSUES',
				issues: validateDocument(result.document),
			});
			setDocumentHandle({
				fileName: result.fileName,
			});
			baselineDocumentRef.current = JSON.stringify(result.document);
			setMessage({
				tone: 'success',
				text: `Imported ${result.fileName}.`,
			});
			requestFitView();
		} catch (error) {
			setMessage({
				tone: 'error',
				text: error instanceof Error ? error.message : 'Open failed.',
			});
		}
	};

	const onSaveDocument = async () => {
		const issues = validateCurrentDocument();
		const errorCount = issues.filter((issue) => issue.severity === 'error').length;
		if (errorCount > 0) {
			setMessage({
				tone: 'error',
				text: `Fix ${errorCount} validation error${errorCount === 1 ? '' : 's'} before exporting.`,
			});
			return;
		}

		try {
			const fileName = documentHandle?.fileName ?? 'evogym-world.json';
			const result = await exportBrowserDocument(state.document, fileName);

			setDocumentHandle({
				fileName: result.fileName,
			});
			baselineDocumentRef.current = serializedDocument;
			setMessage({
				tone: 'success',
				text: `Exported ${result.fileName}.`,
			});
		} catch (error) {
			setMessage({
				tone: 'error',
				text: error instanceof Error ? error.message : 'Save failed.',
			});
		}
	};

	const onValidate = () => {
		validateCurrentDocument();
		setMessage(null);
	};

	return (
		<div className="grid h-full min-h-0 overflow-hidden grid-rows-[auto_1fr_auto]">
			<TopBar
				documentHandle={documentHandle}
				newDocumentDialogOpen={newDocumentDialogOpen}
				onNewDocumentDialogOpenChange={setNewDocumentDialogOpen}
				onNewDocument={onNewDocument}
				onOpenDocument={onOpenDocument}
				onRequestFitView={requestFitView}
				onSaveDocument={onSaveDocument}
				onValidate={onValidate}
			/>
			<div className="grid min-h-0 min-w-0 xl:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
				<section className="grid min-h-0 min-w-0 grid-rows-[auto_auto_1fr] gap-3 p-3 md:p-4 xl:grid-rows-[auto_1fr]">
					<VoxelPalette />
					<Dialog.Root open={mobileControlsOpen} onOpenChange={setMobileControlsOpen}>
						<Dialog.Trigger className="xl:hidden inline-flex items-center justify-center gap-2 rounded-full border border-border bg-surface px-3 py-2 text-sm font-medium text-ink transition hover:border-accent hover:text-accent">
							<FadersHorizontal size={15} />
							Controls
						</Dialog.Trigger>
						<Dialog.Portal>
							<Dialog.Backdrop className="dialog-backdrop fixed inset-0 z-40 bg-ink/30 xl:hidden" />
							<Dialog.Popup className="fixed bottom-[max(0.75rem,env(safe-area-inset-bottom))] right-[max(0.75rem,env(safe-area-inset-right))] top-[max(0.75rem,env(safe-area-inset-top))] z-50 flex w-[min(94vw,24rem)] flex-col overflow-hidden rounded-[1.4rem] border border-border bg-panel shadow-2xl xl:hidden">
								<div className="flex items-center justify-between border-b border-border/70 px-4 py-3">
									<Dialog.Title className="text-sm font-semibold uppercase text-muted">
										Controls
									</Dialog.Title>
									<Dialog.Close
										aria-label="Close controls"
										className="inline-flex size-8 items-center justify-center rounded-full border border-border bg-surface text-muted transition hover:border-accent hover:text-accent"
									>
										<X size={14} />
									</Dialog.Close>
								</div>
								<div className="designer-scroll grid min-h-0 min-w-0 flex-1 gap-4 overflow-y-auto overflow-x-hidden p-4">
									<DesignerPanels
										markerPlacementMode={markerPlacementMode}
										onDeleteDialogOpenChange={setDeleteDialogOpen}
										setMarkerPlacementMode={setMarkerPlacementMode}
									/>
								</div>
							</Dialog.Popup>
						</Dialog.Portal>
					</Dialog.Root>
					<CanvasEditor
						cursor={cursor}
						fitSignal={fitSignal}
						markerPlacementMode={markerPlacementMode}
						setCursor={setCursor}
						setMarkerPlacementMode={setMarkerPlacementMode}
						setView={setView}
						view={view}
					/>
				</section>
				<aside className="designer-scroll hidden min-h-0 min-w-0 auto-rows-max gap-4 overflow-y-auto overflow-x-hidden border-l border-border/80 bg-panel/70 p-4 xl:grid">
					<DesignerPanels
						markerPlacementMode={markerPlacementMode}
						onDeleteDialogOpenChange={setDeleteDialogOpen}
						setMarkerPlacementMode={setMarkerPlacementMode}
					/>
				</aside>
			</div>
			<StatusBar
				cursor={cursor}
				selectedObjectName={selectedObject?.name ?? null}
				view={view}
				onToggleShortcuts={() => setShowShortcuts((value) => !value)}
			/>
			<Toast message={message} />
			<ShortcutSheet open={showShortcuts} onOpenChange={setShowShortcuts} />
		</div>
	);
}

export default function DesignerApp() {
	return (
		<EditorProvider storageKey={browserStorageKey}>
			<DesignerShell />
		</EditorProvider>
	);
}
