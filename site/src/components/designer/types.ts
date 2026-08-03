export type GroupKey = 'materials' | 'actuators';

export type VoxelType =
	| 1
	| 2
	| 3
	| 4
	| 5
	| 6
	| 7
	| 8
	| 9
	| 10
	| 11
	| 12
	| 13
	| 14
	| 15
	| 16
	| 17
	| 18
	| 19
	| 20
	| 21
	| 22
	| 23
	| 24
	| 25
	| 26
	| 27
	| 28
	| 29
	| 30;

export interface GridPoint {
	x: number;
	y: number;
}

export type WorldDirection = 'north' | 'east' | 'south' | 'west';

export interface WorldPoint extends GridPoint {}

export interface WorldMetadata {
	spawn: WorldPoint | null;
	target: WorldPoint | null;
	direction: WorldDirection | null;
	mirrorEnabled: boolean;
}

export interface ViewState {
	scale: number;
	offsetX: number;
	offsetY: number;
}

export interface CursorState extends GridPoint {}

export interface RectPreview {
	start: GridPoint;
	current: GridPoint;
}

export interface VoxelCell extends GridPoint {
	type: VoxelType;
}

export interface Origin extends GridPoint {}

export type EditorConnectivity = Record<string, GridPoint[]>;

export interface EditorObject {
	id: string;
	name: string;
	origin: Origin;
	/** Occupied cells only. Empty voxel type 0 is never stored in the editor model. */
	voxels: VoxelCell[];
	/**
	 * Custom edges keyed by object-local `x:y` coordinates. `null` means export
	 * derives ordinary four-neighbor connectivity from the current geometry.
	 */
	connectivity: EditorConnectivity | null;
	visible: boolean;
	locked: boolean;
}

export interface DocumentPayload {
	gridWidth: number;
	gridHeight: number;
	objects: EditorObject[];
	metadata: WorldMetadata;
}

export type ValidationSeverity = 'error' | 'warning';

export interface ValidationIssue {
	severity: ValidationSeverity;
	code: string;
	message: string;
	objectId?: string | null;
}

export interface HistoryEntry {
	document: DocumentPayload;
	selectedObjectId: string | null;
}

export interface PersistentEditorState {
	document: DocumentPayload;
	selectedObjectId: string | null;
	activeGroup: GroupKey;
	activeTypeInGroup: Record<GroupKey, VoxelType>;
	selectedVoxelType: VoxelType;
	issues: ValidationIssue[];
	history: HistoryEntry[];
	future: HistoryEntry[];
}

export interface WorldJsonObject {
	indices: number[];
	types: number[];
	neighbors: Record<string, number[]>;
}

export interface WorldJsonMetadata {
	spawn: WorldPoint | null;
	target: WorldPoint | null;
	direction: WorldDirection | null;
	mirror_enabled: boolean;
}

export interface WorldJson {
	grid_width: number;
	grid_height: number;
	objects: Record<string, WorldJsonObject>;
	metadata: WorldJsonMetadata;
}

export type EditorAction =
	| { type: 'SET_GROUP'; group: GroupKey }
	| { type: 'SELECT_VOXEL_TYPE'; voxelType: VoxelType }
	| { type: 'CYCLE_TYPE_IN_GROUP' }
	| { type: 'SELECT_OBJECT'; objectId: string | null }
	| { type: 'SET_ISSUES'; issues: ValidationIssue[] }
	| { type: 'NEW_DOCUMENT'; width: number; height: number }
	| { type: 'IMPORT_DOCUMENT'; document: DocumentPayload }
	| { type: 'SET_GRID_SIZE'; width: number; height: number }
	| { type: 'CREATE_OBJECT_AT'; worldX: number; worldY: number }
	| { type: 'PAINT_AT'; worldX: number; worldY: number }
	| { type: 'ERASE_AT'; worldX: number; worldY: number }
	| { type: 'PAINT_RECT'; start: GridPoint; end: GridPoint }
	| { type: 'SET_SPAWN_POINT'; point: WorldPoint }
	| { type: 'SET_TARGET_POINT'; point: WorldPoint }
	| { type: 'CLEAR_SPAWN_POINT' }
	| { type: 'CLEAR_TARGET_POINT' }
	| { type: 'SET_DIRECTION'; direction: WorldDirection | null }
	| { type: 'SET_MIRROR_ENABLED'; enabled: boolean }
	| { type: 'MOVE_OBJECT_BY'; objectId: string; deltaX: number; deltaY: number }
	| { type: 'DELETE_OBJECT'; objectId: string }
	| { type: 'DUPLICATE_OBJECT'; objectId: string }
	| { type: 'ROTATE_OBJECT'; objectId: string }
	| { type: 'MIRROR_OBJECT'; objectId: string }
	| { type: 'RENAME_OBJECT'; objectId: string; name: string }
	| { type: 'TOGGLE_VISIBLE'; objectId: string }
	| { type: 'TOGGLE_LOCKED'; objectId: string }
	| { type: 'UNDO' }
	| { type: 'REDO' };
