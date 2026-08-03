import {
	DEFAULT_GRID_HEIGHT,
	DEFAULT_GRID_WIDTH,
	DEFAULT_TYPE_IN_GROUP,
	HISTORY_LIMIT,
	MULTI_CELL_SLOPE_TYPES,
	VOXEL_GROUP_BY_KEY,
	VOXEL_GROUPS,
} from '../constants';
import type { DocumentPayload, EditorAction, GroupKey, HistoryEntry, PersistentEditorState, VoxelType } from '../types';
import {
	createDefaultMetadata,
	createEmptyDocument,
	cloneDocument,
	createObjectAt,
	deleteObject,
	duplicateObject,
	ensureValidSelection,
	findObjectAt,
	mirrorObject,
	moveObject,
	normalizeWorldPoint,
	normalizeObject,
	objectHasMultiCellSlopeCells,
	paintRect,
	removeVoxelFromObject,
	renameObject,
	rotateObject,
	setGridSize,
	setVoxelOnObject,
	toggleObjectLocked,
	toggleObjectVisibility,
} from '../lib/document';

export const STORAGE_KEY = 'jax-evogym-design-draft';
const DIRECTION_VALUES = ['north', 'east', 'south', 'west'] as const;

function isStoredDocument(value: unknown): value is DocumentPayload {
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return false;
	}

	const record = value as Record<string, unknown>;
	return Number.isInteger(record.gridWidth) && Number.isInteger(record.gridHeight) && Array.isArray(record.objects);
}

function normalizeStoredMetadata(value: unknown): DocumentPayload['metadata'] {
	const defaults = createDefaultMetadata();
	if (!value || typeof value !== 'object' || Array.isArray(value)) {
		return defaults;
	}

	const record = value as Record<string, unknown>;
	const spawn =
		record.spawn &&
		typeof record.spawn === 'object' &&
		!Array.isArray(record.spawn) &&
		Number.isInteger((record.spawn as Record<string, unknown>).x) &&
		Number.isInteger((record.spawn as Record<string, unknown>).y)
			? normalizeWorldPoint({
				x: Number((record.spawn as Record<string, unknown>).x),
				y: Number((record.spawn as Record<string, unknown>).y),
			})
			: null;
	const target =
		record.target &&
		typeof record.target === 'object' &&
		!Array.isArray(record.target) &&
		Number.isInteger((record.target as Record<string, unknown>).x) &&
		Number.isInteger((record.target as Record<string, unknown>).y)
			? normalizeWorldPoint({
				x: Number((record.target as Record<string, unknown>).x),
				y: Number((record.target as Record<string, unknown>).y),
			})
			: null;
	const direction =
		typeof record.direction === 'string' &&
		DIRECTION_VALUES.includes(record.direction as (typeof DIRECTION_VALUES)[number])
			? (record.direction as DocumentPayload['metadata']['direction'])
			: null;

	return {
		spawn,
		target,
		direction,
		mirrorEnabled: record.mirrorEnabled === true,
	};
}

function normalizeStoredDocument(document: DocumentPayload): DocumentPayload {
	const normalized = cloneDocument({
		...document,
		metadata: normalizeStoredMetadata((document as { metadata?: unknown }).metadata),
	});
	const sized = setGridSize(normalized, normalized.gridWidth, normalized.gridHeight);
	for (const object of sized.objects) {
		normalizeObject(object);
	}
	return sized;
}

function getStoredDocument(storageKey: string) {
	if (typeof localStorage === 'undefined') {
		return null;
	}

	try {
		const raw = localStorage.getItem(storageKey);
		if (!raw) {
			return null;
		}

		const parsed = JSON.parse(raw) as unknown;
		return isStoredDocument(parsed) ? normalizeStoredDocument(parsed) : null;
	} catch {
		return null;
	}
}

function snapshot(state: PersistentEditorState): HistoryEntry {
	return {
		document: state.document,
		selectedObjectId: state.selectedObjectId,
	};
}

function pushHistory(state: PersistentEditorState, nextDocument: DocumentPayload, selectedObjectId: string | null) {
	const nextHistory = [...state.history, snapshot(state)];
	if (nextHistory.length > HISTORY_LIMIT) {
		nextHistory.shift();
	}

	return {
		...state,
		document: nextDocument,
		selectedObjectId: ensureValidSelection(nextDocument, selectedObjectId),
		history: nextHistory,
		future: [],
	};
}

function commitDocumentChange(
	state: PersistentEditorState,
	nextDocument: DocumentPayload,
	selectedObjectId = state.selectedObjectId,
) {
	if (nextDocument === state.document) {
		return state;
	}

	return pushHistory(state, nextDocument, selectedObjectId);
}

function withIssue(
	state: PersistentEditorState,
	code: string,
	message: string,
	objectId?: string | null,
) {
	return {
		...state,
		issues: [
			...state.issues.filter((issue) => issue.code !== code),
			{
				severity: 'warning' as const,
				code,
				message,
				objectId: objectId ?? null,
			},
		],
	};
}

function nextTypeInGroup(group: GroupKey, currentType: VoxelType): VoxelType {
	const types = VOXEL_GROUP_BY_KEY[group].types;
	const index = types.indexOf(currentType);
	return types[(index + 1) % types.length];
}

export function createInitialEditorState(storageKey = STORAGE_KEY): PersistentEditorState {
	const document =
		getStoredDocument(storageKey) ?? createEmptyDocument(DEFAULT_GRID_WIDTH, DEFAULT_GRID_HEIGHT);
	const activeGroup: GroupKey = 'materials';
	const activeTypeInGroup = { ...DEFAULT_TYPE_IN_GROUP };

	return {
		document,
		selectedObjectId: ensureValidSelection(document, null),
		activeGroup,
		activeTypeInGroup,
		selectedVoxelType: activeTypeInGroup[activeGroup],
		issues: [],
		history: [],
		future: [],
	};
}

export function editorReducer(state: PersistentEditorState, action: EditorAction): PersistentEditorState {
	switch (action.type) {
		case 'SET_GROUP': {
			const nextGroup = action.group;
			const nextType = state.activeTypeInGroup[nextGroup];
			return {
				...state,
				activeGroup: nextGroup,
				selectedVoxelType: nextType,
			};
		}

		case 'SELECT_VOXEL_TYPE': {
			const nextType = action.voxelType;
			const owningGroup = VOXEL_GROUPS.find((group) => group.types.includes(nextType));
			if (!owningGroup) {
				return state;
			}
			return {
				...state,
				activeGroup: owningGroup.key,
				activeTypeInGroup: {
					...state.activeTypeInGroup,
					[owningGroup.key]: nextType,
				},
				selectedVoxelType: nextType,
			};
		}

		case 'CYCLE_TYPE_IN_GROUP': {
			const nextType = nextTypeInGroup(state.activeGroup, state.activeTypeInGroup[state.activeGroup]);
			return {
				...state,
				activeTypeInGroup: {
					...state.activeTypeInGroup,
					[state.activeGroup]: nextType,
				},
				selectedVoxelType: nextType,
			};
		}

		case 'SELECT_OBJECT':
			return {
				...state,
				selectedObjectId: ensureValidSelection(state.document, action.objectId),
			};

		case 'SET_ISSUES':
			return {
				...state,
				issues: action.issues,
			};

		case 'NEW_DOCUMENT': {
			const document = createEmptyDocument(action.width, action.height);
			return {
				...state,
				document,
				selectedObjectId: null,
				issues: [],
				history: [],
				future: [],
			};
		}

		case 'IMPORT_DOCUMENT': {
			const document = normalizeStoredDocument(action.document);
			return {
				...state,
				document,
				selectedObjectId: ensureValidSelection(document, null),
				issues: [],
				history: [],
				future: [],
			};
		}

		case 'SET_GRID_SIZE':
			return commitDocumentChange(
				state,
				setGridSize(state.document, action.width, action.height),
			);

		case 'CREATE_OBJECT_AT': {
			const result = createObjectAt(
				state.document,
				action.worldX,
				action.worldY,
				state.selectedVoxelType,
			);
			return commitDocumentChange(state, result.document, result.objectId);
		}

		case 'PAINT_AT': {
			const selectedObject = state.selectedObjectId
				? state.document.objects.find((object) => object.id === state.selectedObjectId)
				: null;
			const occupant = findObjectAt(
				state.document,
				action.worldX,
				action.worldY,
				{ visibleOnly: false },
			);
			if (occupant && occupant.id !== selectedObject?.id) {
				return state;
			}

			if (!selectedObject || selectedObject.locked) {
				if (occupant) {
					return state;
				}
				const result = createObjectAt(
					state.document,
					action.worldX,
					action.worldY,
					state.selectedVoxelType,
				);
				return commitDocumentChange(state, result.document, result.objectId);
			}

			return commitDocumentChange(
				state,
				setVoxelOnObject(
					state.document,
					selectedObject.id,
					action.worldX,
					action.worldY,
					state.selectedVoxelType,
				),
			);
		}

		case 'ERASE_AT': {
			const target = findObjectAt(state.document, action.worldX, action.worldY, { visibleOnly: false });
			if (!target || target.locked) {
				return state;
			}

			const nextDocument = removeVoxelFromObject(state.document, target.id, action.worldX, action.worldY);
			return commitDocumentChange(
				state,
				nextDocument,
				ensureValidSelection(nextDocument, state.selectedObjectId),
			);
		}

		case 'PAINT_RECT': {
			if (MULTI_CELL_SLOPE_TYPES.has(state.selectedVoxelType)) {
				return withIssue(
					state,
					'slope2_rect_disabled',
					'Rectangle fill is disabled for 2x1/3x1 slope blocks.',
					state.selectedObjectId,
				);
			}
			const selectedObject = state.selectedObjectId
				? state.document.objects.find((object) => object.id === state.selectedObjectId)
				: null;

			if (!selectedObject || selectedObject.locked) {
				const created = createObjectAt(
					state.document,
					action.start.x,
					action.start.y,
					state.selectedVoxelType,
				);
				const nextDocument = paintRect(
					created.document,
					created.objectId,
					action.start,
					action.end,
					state.selectedVoxelType,
				);
				return commitDocumentChange(state, nextDocument, created.objectId);
			}

			return commitDocumentChange(
				state,
				paintRect(
					state.document,
					selectedObject.id,
					action.start,
					action.end,
					state.selectedVoxelType,
				),
			);
		}

		case 'SET_SPAWN_POINT': {
			const point = normalizeWorldPoint(action.point);
			const current = state.document.metadata.spawn;
			if (current && current.x === point.x && current.y === point.y) {
				return state;
			}
			return commitDocumentChange(state, {
				...state.document,
				metadata: {
					...state.document.metadata,
					spawn: point,
				},
			});
		}

		case 'SET_TARGET_POINT': {
			const point = normalizeWorldPoint(action.point);
			const current = state.document.metadata.target;
			if (current && current.x === point.x && current.y === point.y) {
				return state;
			}
			return commitDocumentChange(state, {
				...state.document,
				metadata: {
					...state.document.metadata,
					target: point,
				},
			});
		}

		case 'CLEAR_SPAWN_POINT':
			if (!state.document.metadata.spawn) {
				return state;
			}
			return commitDocumentChange(state, {
				...state.document,
				metadata: {
					...state.document.metadata,
					spawn: null,
				},
			});

		case 'CLEAR_TARGET_POINT':
			if (!state.document.metadata.target) {
				return state;
			}
			return commitDocumentChange(state, {
				...state.document,
				metadata: {
					...state.document.metadata,
					target: null,
				},
			});

		case 'SET_DIRECTION':
			if (state.document.metadata.direction === action.direction) {
				return state;
			}
			return commitDocumentChange(state, {
				...state.document,
				metadata: {
					...state.document.metadata,
					direction: action.direction,
				},
			});

		case 'SET_MIRROR_ENABLED':
			if (state.document.metadata.mirrorEnabled === action.enabled) {
				return state;
			}
			return commitDocumentChange(state, {
				...state.document,
				metadata: {
					...state.document.metadata,
					mirrorEnabled: action.enabled,
				},
			});

		case 'MOVE_OBJECT_BY':
			return commitDocumentChange(
				state,
				moveObject(state.document, action.objectId, action.deltaX, action.deltaY),
				action.objectId,
			);

		case 'DELETE_OBJECT': {
			const nextDocument = deleteObject(state.document, action.objectId);
			return commitDocumentChange(
				state,
				nextDocument,
				ensureValidSelection(nextDocument, state.selectedObjectId),
			);
		}

		case 'DUPLICATE_OBJECT': {
			const result = duplicateObject(state.document, action.objectId);
			return commitDocumentChange(state, result.document, result.objectId);
		}

		case 'ROTATE_OBJECT': {
			const target = state.document.objects.find((object) => object.id === action.objectId) ?? null;
			if (target && objectHasMultiCellSlopeCells(target)) {
				return withIssue(
					state,
					'slope2_rotate_disabled',
					`Object '${target.name}' contains 2x1/3x1 slope blocks and cannot be rotated in this release.`,
					target.id,
				);
			}
			return commitDocumentChange(
				state,
				rotateObject(state.document, action.objectId),
				action.objectId,
			);
		}

		case 'MIRROR_OBJECT':
			return commitDocumentChange(
				state,
				mirrorObject(state.document, action.objectId),
				action.objectId,
			);

		case 'RENAME_OBJECT':
			return commitDocumentChange(
				state,
				renameObject(state.document, action.objectId, action.name),
				action.objectId,
			);

		case 'TOGGLE_VISIBLE':
			return commitDocumentChange(
				state,
				toggleObjectVisibility(state.document, action.objectId),
				action.objectId,
			);

		case 'TOGGLE_LOCKED':
			return commitDocumentChange(
				state,
				toggleObjectLocked(state.document, action.objectId),
				action.objectId,
			);

		case 'UNDO': {
			const previous = state.history.at(-1);
			if (!previous) {
				return state;
			}

			return {
				...state,
				document: previous.document,
				selectedObjectId: ensureValidSelection(previous.document, previous.selectedObjectId),
				history: state.history.slice(0, -1),
				future: [snapshot(state), ...state.future],
			};
		}

		case 'REDO': {
			const [next, ...future] = state.future;
			if (!next) {
				return state;
			}

			return {
				...state,
				document: next.document,
				selectedObjectId: ensureValidSelection(next.document, next.selectedObjectId),
				history: [...state.history, snapshot(state)].slice(-HISTORY_LIMIT),
				future,
			};
		}

		default:
			return state;
	}
}
