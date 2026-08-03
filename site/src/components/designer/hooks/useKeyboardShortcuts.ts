import { useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';

import { GROUP_HOTKEYS, VOXEL_GROUPS } from '../constants';
import { findObjectAt } from '../lib/document';
import type { CursorState, EditorAction, GroupKey, PersistentEditorState } from '../types';

function isEditableTarget(target: EventTarget | null) {
	return (
		target instanceof HTMLInputElement ||
		target instanceof HTMLTextAreaElement ||
		target instanceof HTMLSelectElement ||
		(target instanceof HTMLElement && target.isContentEditable)
	);
}

const GROUP_KEYS: GroupKey[] = VOXEL_GROUPS.map((group) => group.key);
export function useKeyboardShortcuts(
	state: PersistentEditorState,
	dispatch: Dispatch<EditorAction>,
	cursor: CursorState | null,
	setShowShortcuts: Dispatch<SetStateAction<boolean>>,
	isModalOpen: boolean,
) {
	useEffect(() => {
		const onKeyDown = (event: KeyboardEvent) => {
			if (event.defaultPrevented || isEditableTarget(event.target) || isModalOpen) {
				return;
			}

			const mod = event.metaKey || event.ctrlKey;
			const key = event.key.toLowerCase();

			// Undo / Redo
			if (mod && key === 'z') {
				event.preventDefault();
				dispatch({ type: event.shiftKey ? 'REDO' : 'UNDO' });
				return;
			}
			if (mod && key === 'y') {
				event.preventDefault();
				dispatch({ type: 'REDO' });
				return;
			}

			// Object operations
			if (!mod && key === 'r' && state.selectedObjectId) {
				event.preventDefault();
				dispatch({ type: 'ROTATE_OBJECT', objectId: state.selectedObjectId });
				return;
			}
			if (mod && key === 'm' && state.selectedObjectId) {
				event.preventDefault();
				dispatch({ type: 'MIRROR_OBJECT', objectId: state.selectedObjectId });
				return;
			}
			if (mod && key === 'd' && state.selectedObjectId) {
				event.preventDefault();
				dispatch({ type: 'DUPLICATE_OBJECT', objectId: state.selectedObjectId });
				return;
			}

			// Delete selected object
			if ((event.key === 'Delete' || event.key === 'Backspace') && state.selectedObjectId && !mod) {
				dispatch({ type: 'DELETE_OBJECT', objectId: state.selectedObjectId });
				return;
			}

			// Group selection: Q / W
			if (!mod) {
				const hotkeyIndex = GROUP_HOTKEYS.indexOf(key as (typeof GROUP_HOTKEYS)[number]);
				if (hotkeyIndex !== -1 && hotkeyIndex < GROUP_KEYS.length) {
					dispatch({ type: 'SET_GROUP', group: GROUP_KEYS[hotkeyIndex] });
					return;
				}
			}

			// Group selection: number keys
			const selectedGroup = Number(event.key);
			if (Number.isInteger(selectedGroup) && selectedGroup >= 1 && selectedGroup <= GROUP_KEYS.length && !mod) {
				const index = selectedGroup - 1;
				dispatch({ type: 'SET_GROUP', group: GROUP_KEYS[index] });
				return;
			}

			// Cycle between groups: A / D
			if ((key === 'a' || key === 'd') && !mod) {
				event.preventDefault();
				const currentIndex = GROUP_KEYS.indexOf(state.activeGroup);
				const delta = key === 'd' ? 1 : -1;
				const nextIndex = (currentIndex + delta + GROUP_KEYS.length) % GROUP_KEYS.length;
				dispatch({ type: 'SET_GROUP', group: GROUP_KEYS[nextIndex] });
				return;
			}

			// Shift + ArrowUp / ArrowDown: alternate cycle between groups
			if (!mod && event.shiftKey && (key === 'arrowup' || key === 'arrowdown')) {
				event.preventDefault();
				const currentIndex = GROUP_KEYS.indexOf(state.activeGroup);
				const delta = key === 'arrowdown' ? 1 : -1;
				const nextIndex = (currentIndex + delta + GROUP_KEYS.length) % GROUP_KEYS.length;
				dispatch({ type: 'SET_GROUP', group: GROUP_KEYS[nextIndex] });
				return;
			}

			// Move selected object with arrow keys
			if (!mod && state.selectedObjectId && (key === 'arrowleft' || key === 'arrowright' || key === 'arrowup' || key === 'arrowdown')) {
				event.preventDefault();
				const deltaX = key === 'arrowleft' ? -1 : key === 'arrowright' ? 1 : 0;
				const deltaY = key === 'arrowup' ? 1 : key === 'arrowdown' ? -1 : 0;
				dispatch({
					type: 'MOVE_OBJECT_BY',
					objectId: state.selectedObjectId,
					deltaX,
					deltaY,
				});
				return;
			}

			// Cycle type within group: S
			if (key === 's' && !mod) {
				dispatch({ type: 'CYCLE_TYPE_IN_GROUP' });
				return;
			}

			// Place voxel at cursor: Space
			if (event.key === ' ' && !mod) {
				event.preventDefault();
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

			// Erase voxel at cursor: X
			if (key === 'x' && !mod) {
				if (cursor) {
					dispatch({ type: 'ERASE_AT', worldX: cursor.x, worldY: cursor.y });
				}
				return;
			}

			// Toggle shortcut sheet: ?
			if (event.key === '?') {
				setShowShortcuts((value) => !value);
				return;
			}
		};

		window.addEventListener('keydown', onKeyDown);
		return () => window.removeEventListener('keydown', onKeyDown);
	}, [
		cursor,
		dispatch,
		isModalOpen,
		setShowShortcuts,
		state.activeGroup,
		state.document,
		state.selectedObjectId,
	]);
}
