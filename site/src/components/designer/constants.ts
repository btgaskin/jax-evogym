import tokens from './visual-tokens.json';
import type { GroupKey, ViewState, VoxelType } from './types';

export type ThemeMode = keyof typeof tokens.themes;
export type DesignThemeTokens = (typeof tokens.themes)[ThemeMode];

export const RIGID = 1;
export const SOFT = 2;
export const H_ACT = 3;
export const V_ACT = 4;
export const FIXED = 5;
export const CONTRACTILE = 6;
export const SLOPE_UP_RIGHT = 7;
export const SLOPE_UP_LEFT = 8;
export const SLOPE_DOWN_RIGHT = 9;
export const SLOPE_DOWN_LEFT = 10;
export const SLOPE2_UP_RIGHT_LIGHT = 11;
export const SLOPE2_UP_RIGHT_HEAVY = 12;
export const SLOPE2_UP_LEFT_HEAVY = 13;
export const SLOPE2_UP_LEFT_LIGHT = 14;
export const SLOPE2_DOWN_RIGHT_LIGHT = 15;
export const SLOPE2_DOWN_RIGHT_HEAVY = 16;
export const SLOPE2_DOWN_LEFT_HEAVY = 17;
export const SLOPE2_DOWN_LEFT_LIGHT = 18;
export const SLOPE3_UP_RIGHT_LIGHT = 19;
export const SLOPE3_UP_RIGHT_MID = 20;
export const SLOPE3_UP_RIGHT_HEAVY = 21;
export const SLOPE3_UP_LEFT_HEAVY = 22;
export const SLOPE3_UP_LEFT_MID = 23;
export const SLOPE3_UP_LEFT_LIGHT = 24;
export const SLOPE3_DOWN_RIGHT_HEAVY = 25;
export const SLOPE3_DOWN_RIGHT_MID = 26;
export const SLOPE3_DOWN_RIGHT_LIGHT = 27;
export const SLOPE3_DOWN_LEFT_LIGHT = 28;
export const SLOPE3_DOWN_LEFT_MID = 29;
export const SLOPE3_DOWN_LEFT_HEAVY = 30;

export const DEFAULT_GRID_WIDTH = 24;
export const DEFAULT_GRID_HEIGHT = 16;
export const MAX_GRID_DIM = 256;
export const DEFAULT_CELL_SIZE = 28;
export const DEFAULT_VIEW: ViewState = {
	scale: 1,
	offsetX: 0,
	offsetY: 0,
};

export const CANVAS_PADDING = 44;
export const HISTORY_LIMIT = 80;

export const ACTUATOR_TYPES = new Set<VoxelType>([H_ACT, V_ACT, CONTRACTILE]);
export const SLOPE_TYPES = new Set<VoxelType>([
	SLOPE_UP_RIGHT,
	SLOPE_UP_LEFT,
	SLOPE_DOWN_RIGHT,
	SLOPE_DOWN_LEFT,
	SLOPE2_UP_RIGHT_LIGHT,
	SLOPE2_UP_RIGHT_HEAVY,
	SLOPE2_UP_LEFT_HEAVY,
	SLOPE2_UP_LEFT_LIGHT,
	SLOPE2_DOWN_RIGHT_LIGHT,
	SLOPE2_DOWN_RIGHT_HEAVY,
	SLOPE2_DOWN_LEFT_HEAVY,
	SLOPE2_DOWN_LEFT_LIGHT,
	SLOPE3_UP_RIGHT_LIGHT,
	SLOPE3_UP_RIGHT_MID,
	SLOPE3_UP_RIGHT_HEAVY,
	SLOPE3_UP_LEFT_HEAVY,
	SLOPE3_UP_LEFT_MID,
	SLOPE3_UP_LEFT_LIGHT,
	SLOPE3_DOWN_RIGHT_HEAVY,
	SLOPE3_DOWN_RIGHT_MID,
	SLOPE3_DOWN_RIGHT_LIGHT,
	SLOPE3_DOWN_LEFT_LIGHT,
	SLOPE3_DOWN_LEFT_MID,
	SLOPE3_DOWN_LEFT_HEAVY,
]);
export const STATIC_TERRAIN_TYPES = new Set<VoxelType>([FIXED, ...SLOPE_TYPES]);
export const SLOPE_2X1_TYPES = new Set<VoxelType>([
	SLOPE2_UP_RIGHT_LIGHT,
	SLOPE2_UP_RIGHT_HEAVY,
	SLOPE2_UP_LEFT_HEAVY,
	SLOPE2_UP_LEFT_LIGHT,
	SLOPE2_DOWN_RIGHT_LIGHT,
	SLOPE2_DOWN_RIGHT_HEAVY,
	SLOPE2_DOWN_LEFT_HEAVY,
	SLOPE2_DOWN_LEFT_LIGHT,
]);
export const SLOPE_2X1_HEAVY_TYPES = new Set<VoxelType>([
	SLOPE2_UP_RIGHT_HEAVY,
	SLOPE2_UP_LEFT_HEAVY,
	SLOPE2_DOWN_RIGHT_HEAVY,
	SLOPE2_DOWN_LEFT_HEAVY,
]);
export const SLOPE_3X1_TYPES = new Set<VoxelType>([
	SLOPE3_UP_RIGHT_LIGHT,
	SLOPE3_UP_RIGHT_MID,
	SLOPE3_UP_RIGHT_HEAVY,
	SLOPE3_UP_LEFT_HEAVY,
	SLOPE3_UP_LEFT_MID,
	SLOPE3_UP_LEFT_LIGHT,
	SLOPE3_DOWN_RIGHT_HEAVY,
	SLOPE3_DOWN_RIGHT_MID,
	SLOPE3_DOWN_RIGHT_LIGHT,
	SLOPE3_DOWN_LEFT_LIGHT,
	SLOPE3_DOWN_LEFT_MID,
	SLOPE3_DOWN_LEFT_HEAVY,
]);
export const SLOPE_3X1_HEAVY_TYPES = new Set<VoxelType>([
	SLOPE3_UP_RIGHT_HEAVY,
	SLOPE3_UP_LEFT_HEAVY,
	SLOPE3_DOWN_RIGHT_HEAVY,
	SLOPE3_DOWN_LEFT_HEAVY,
]);
export const MULTI_CELL_SLOPE_TYPES = new Set<VoxelType>([
	...SLOPE_2X1_TYPES,
	...SLOPE_3X1_TYPES,
]);

export const SLOPE_2X1_MATE_TYPE: Partial<Record<VoxelType, VoxelType>> = {
	[SLOPE2_UP_RIGHT_LIGHT]: SLOPE2_UP_RIGHT_HEAVY,
	[SLOPE2_UP_RIGHT_HEAVY]: SLOPE2_UP_RIGHT_LIGHT,
	[SLOPE2_UP_LEFT_HEAVY]: SLOPE2_UP_LEFT_LIGHT,
	[SLOPE2_UP_LEFT_LIGHT]: SLOPE2_UP_LEFT_HEAVY,
	[SLOPE2_DOWN_RIGHT_LIGHT]: SLOPE2_DOWN_RIGHT_HEAVY,
	[SLOPE2_DOWN_RIGHT_HEAVY]: SLOPE2_DOWN_RIGHT_LIGHT,
	[SLOPE2_DOWN_LEFT_HEAVY]: SLOPE2_DOWN_LEFT_LIGHT,
	[SLOPE2_DOWN_LEFT_LIGHT]: SLOPE2_DOWN_LEFT_HEAVY,
};

export const SLOPE_2X1_MATE_OFFSET: Partial<Record<VoxelType, [number, number]>> = {
	[SLOPE2_UP_RIGHT_LIGHT]: [1, 0],
	[SLOPE2_UP_RIGHT_HEAVY]: [-1, 0],
	[SLOPE2_UP_LEFT_HEAVY]: [1, 0],
	[SLOPE2_UP_LEFT_LIGHT]: [-1, 0],
	[SLOPE2_DOWN_RIGHT_LIGHT]: [-1, 0],
	[SLOPE2_DOWN_RIGHT_HEAVY]: [1, 0],
	[SLOPE2_DOWN_LEFT_HEAVY]: [-1, 0],
	[SLOPE2_DOWN_LEFT_LIGHT]: [1, 0],
};
export const SLOPE_3X1_NEIGHBOR_RULES: Partial<Record<VoxelType, Array<[number, number, VoxelType]>>> = {
	[SLOPE3_UP_RIGHT_LIGHT]: [[1, 0, SLOPE3_UP_RIGHT_MID], [2, 0, SLOPE3_UP_RIGHT_HEAVY]],
	[SLOPE3_UP_RIGHT_MID]: [[-1, 0, SLOPE3_UP_RIGHT_LIGHT], [1, 0, SLOPE3_UP_RIGHT_HEAVY]],
	[SLOPE3_UP_RIGHT_HEAVY]: [[-1, 0, SLOPE3_UP_RIGHT_MID], [-2, 0, SLOPE3_UP_RIGHT_LIGHT]],
	[SLOPE3_UP_LEFT_HEAVY]: [[1, 0, SLOPE3_UP_LEFT_MID], [2, 0, SLOPE3_UP_LEFT_LIGHT]],
	[SLOPE3_UP_LEFT_MID]: [[-1, 0, SLOPE3_UP_LEFT_HEAVY], [1, 0, SLOPE3_UP_LEFT_LIGHT]],
	[SLOPE3_UP_LEFT_LIGHT]: [[-1, 0, SLOPE3_UP_LEFT_MID], [-2, 0, SLOPE3_UP_LEFT_HEAVY]],
	[SLOPE3_DOWN_RIGHT_HEAVY]: [[1, 0, SLOPE3_DOWN_RIGHT_MID], [2, 0, SLOPE3_DOWN_RIGHT_LIGHT]],
	[SLOPE3_DOWN_RIGHT_MID]: [[-1, 0, SLOPE3_DOWN_RIGHT_HEAVY], [1, 0, SLOPE3_DOWN_RIGHT_LIGHT]],
	[SLOPE3_DOWN_RIGHT_LIGHT]: [[-1, 0, SLOPE3_DOWN_RIGHT_MID], [-2, 0, SLOPE3_DOWN_RIGHT_HEAVY]],
	[SLOPE3_DOWN_LEFT_LIGHT]: [[1, 0, SLOPE3_DOWN_LEFT_MID], [2, 0, SLOPE3_DOWN_LEFT_HEAVY]],
	[SLOPE3_DOWN_LEFT_MID]: [[-1, 0, SLOPE3_DOWN_LEFT_LIGHT], [1, 0, SLOPE3_DOWN_LEFT_HEAVY]],
	[SLOPE3_DOWN_LEFT_HEAVY]: [[-1, 0, SLOPE3_DOWN_LEFT_MID], [-2, 0, SLOPE3_DOWN_LEFT_LIGHT]],
};
export const SLOPE_3X1_HEAVY_ANCHOR_TYPE: Partial<Record<VoxelType, VoxelType>> = {
	[SLOPE3_UP_RIGHT_LIGHT]: SLOPE3_UP_RIGHT_HEAVY,
	[SLOPE3_UP_RIGHT_MID]: SLOPE3_UP_RIGHT_HEAVY,
	[SLOPE3_UP_RIGHT_HEAVY]: SLOPE3_UP_RIGHT_HEAVY,
	[SLOPE3_UP_LEFT_HEAVY]: SLOPE3_UP_LEFT_HEAVY,
	[SLOPE3_UP_LEFT_MID]: SLOPE3_UP_LEFT_HEAVY,
	[SLOPE3_UP_LEFT_LIGHT]: SLOPE3_UP_LEFT_HEAVY,
	[SLOPE3_DOWN_RIGHT_HEAVY]: SLOPE3_DOWN_RIGHT_HEAVY,
	[SLOPE3_DOWN_RIGHT_MID]: SLOPE3_DOWN_RIGHT_HEAVY,
	[SLOPE3_DOWN_RIGHT_LIGHT]: SLOPE3_DOWN_RIGHT_HEAVY,
	[SLOPE3_DOWN_LEFT_LIGHT]: SLOPE3_DOWN_LEFT_HEAVY,
	[SLOPE3_DOWN_LEFT_MID]: SLOPE3_DOWN_LEFT_HEAVY,
	[SLOPE3_DOWN_LEFT_HEAVY]: SLOPE3_DOWN_LEFT_HEAVY,
};
export const SLOPE_3X1_SEGMENTS_FROM_HEAVY: Partial<Record<VoxelType, Array<[number, number, VoxelType]>>> = {
	[SLOPE3_UP_RIGHT_HEAVY]: [[0, 0, SLOPE3_UP_RIGHT_HEAVY], [-1, 0, SLOPE3_UP_RIGHT_MID], [-2, 0, SLOPE3_UP_RIGHT_LIGHT]],
	[SLOPE3_UP_LEFT_HEAVY]: [[0, 0, SLOPE3_UP_LEFT_HEAVY], [1, 0, SLOPE3_UP_LEFT_MID], [2, 0, SLOPE3_UP_LEFT_LIGHT]],
	[SLOPE3_DOWN_RIGHT_HEAVY]: [[0, 0, SLOPE3_DOWN_RIGHT_HEAVY], [1, 0, SLOPE3_DOWN_RIGHT_MID], [2, 0, SLOPE3_DOWN_RIGHT_LIGHT]],
	[SLOPE3_DOWN_LEFT_HEAVY]: [[0, 0, SLOPE3_DOWN_LEFT_HEAVY], [-1, 0, SLOPE3_DOWN_LEFT_MID], [-2, 0, SLOPE3_DOWN_LEFT_LIGHT]],
};

/* ── Palette groups ─────────────────────────────────── */

export interface VoxelGroupDef {
	key: GroupKey;
	label: string;
	types: VoxelType[];
}

export const VOXEL_GROUPS: VoxelGroupDef[] = [
	{ key: 'materials', label: 'Materials', types: [RIGID, SOFT, FIXED] },
	{ key: 'actuators', label: 'Actuators', types: [H_ACT, V_ACT, CONTRACTILE] },
];

export const GROUP_HOTKEYS = ['q', 'w'] as const;

export const VOXEL_GROUP_BY_KEY = Object.fromEntries(
	VOXEL_GROUPS.map((group) => [group.key, group]),
) as Record<GroupKey, VoxelGroupDef>;

export const DEFAULT_TYPE_IN_GROUP: Record<GroupKey, VoxelType> = {
	materials: RIGID,
	actuators: H_ACT,
};

/** SVG polygon points (12x12 viewBox, y-down) for each slope orientation. */
export const SLOPE_SWATCH_POINTS: Partial<Record<VoxelType, string>> = {
	[SLOPE_UP_RIGHT]: '0,12 12,12 12,0',
	[SLOPE_UP_LEFT]: '0,12 12,12 0,0',
	[SLOPE_DOWN_RIGHT]: '0,12 0,0 12,0',
	[SLOPE_DOWN_LEFT]: '12,12 12,0 0,0',
	[SLOPE2_UP_RIGHT_HEAVY]: '0,12 12,12 12,0 0,6',
	[SLOPE2_UP_LEFT_HEAVY]: '0,12 12,12 12,6 0,0',
	[SLOPE2_DOWN_RIGHT_HEAVY]: '0,12 12,6 12,0 0,0',
	[SLOPE2_DOWN_LEFT_HEAVY]: '0,6 12,12 12,0 0,0',
	[SLOPE2_UP_RIGHT_LIGHT]: '0,12 12,12 12,6',
	[SLOPE2_UP_LEFT_LIGHT]: '0,12 12,12 0,6',
	[SLOPE2_DOWN_RIGHT_LIGHT]: '0,6 12,0 0,0',
	[SLOPE2_DOWN_LEFT_LIGHT]: '0,0 12,6 12,0',
	[SLOPE3_UP_RIGHT_LIGHT]: '0,12 12,12 12,8',
	[SLOPE3_UP_RIGHT_MID]: '0,12 12,12 12,4 0,8',
	[SLOPE3_UP_RIGHT_HEAVY]: '0,12 12,12 12,0 0,4',
	[SLOPE3_UP_LEFT_HEAVY]: '0,12 12,12 12,4 0,0',
	[SLOPE3_UP_LEFT_MID]: '0,12 12,12 12,8 0,4',
	[SLOPE3_UP_LEFT_LIGHT]: '0,12 12,12 0,8',
	[SLOPE3_DOWN_RIGHT_HEAVY]: '0,12 12,8 12,0 0,0',
	[SLOPE3_DOWN_RIGHT_MID]: '0,8 12,4 12,0 0,0',
	[SLOPE3_DOWN_RIGHT_LIGHT]: '0,4 12,0 0,0',
	[SLOPE3_DOWN_LEFT_LIGHT]: '0,0 12,4 12,0',
	[SLOPE3_DOWN_LEFT_MID]: '0,4 12,8 12,0 0,0',
	[SLOPE3_DOWN_LEFT_HEAVY]: '0,8 12,12 12,0 0,0',
};

/* ── Voxel visual definitions ───────────────────────── */

export interface VoxelDefinition {
	type: VoxelType;
	key: string;
	label: string;
	kind: 'solid' | 'gradient';
	hex?: string;
	start?: string;
	end?: string;
}

export const VOXEL_DEFINITIONS: VoxelDefinition[] = [
	{ type: RIGID, key: 'rigid', label: 'Rigid', kind: 'solid', hex: tokens.voxels.rigid.hex },
	{ type: SOFT, key: 'soft', label: 'Soft', kind: 'solid', hex: tokens.voxels.soft.hex },
	{ type: H_ACT, key: 'h_act', label: 'H-Act', kind: 'gradient', start: tokens.voxels.h_act.start, end: tokens.voxels.h_act.end },
	{ type: V_ACT, key: 'v_act', label: 'V-Act', kind: 'gradient', start: tokens.voxels.v_act.start, end: tokens.voxels.v_act.end },
	{ type: FIXED, key: 'fixed', label: 'Fixed', kind: 'solid', hex: tokens.voxels.fixed.hex },
	{ type: CONTRACTILE, key: 'contractile', label: 'Contractile', kind: 'gradient', start: tokens.voxels.contractile.start, end: tokens.voxels.contractile.end },
	{ type: SLOPE_UP_RIGHT, key: 'slope_up_right', label: 'UR', kind: 'solid', hex: tokens.voxels.slope_up_right.hex },
	{ type: SLOPE_UP_LEFT, key: 'slope_up_left', label: 'UL', kind: 'solid', hex: tokens.voxels.slope_up_left.hex },
	{ type: SLOPE_DOWN_RIGHT, key: 'slope_down_right', label: 'DR', kind: 'solid', hex: tokens.voxels.slope_down_right.hex },
	{ type: SLOPE_DOWN_LEFT, key: 'slope_down_left', label: 'DL', kind: 'solid', hex: tokens.voxels.slope_down_left.hex },
	{ type: SLOPE2_UP_RIGHT_LIGHT, key: 'slope2_up_right_light', label: '2x1 UR Light', kind: 'solid', hex: tokens.voxels.slope2_up_right_light.hex },
	{ type: SLOPE2_UP_RIGHT_HEAVY, key: 'slope2_up_right_heavy', label: '2x1 UR', kind: 'solid', hex: tokens.voxels.slope2_up_right_heavy.hex },
	{ type: SLOPE2_UP_LEFT_HEAVY, key: 'slope2_up_left_heavy', label: '2x1 UL', kind: 'solid', hex: tokens.voxels.slope2_up_left_heavy.hex },
	{ type: SLOPE2_UP_LEFT_LIGHT, key: 'slope2_up_left_light', label: '2x1 UL Light', kind: 'solid', hex: tokens.voxels.slope2_up_left_light.hex },
	{ type: SLOPE2_DOWN_RIGHT_LIGHT, key: 'slope2_down_right_light', label: '2x1 DR Light', kind: 'solid', hex: tokens.voxels.slope2_down_right_light.hex },
	{ type: SLOPE2_DOWN_RIGHT_HEAVY, key: 'slope2_down_right_heavy', label: '2x1 DR', kind: 'solid', hex: tokens.voxels.slope2_down_right_heavy.hex },
	{ type: SLOPE2_DOWN_LEFT_HEAVY, key: 'slope2_down_left_heavy', label: '2x1 DL', kind: 'solid', hex: tokens.voxels.slope2_down_left_heavy.hex },
	{ type: SLOPE2_DOWN_LEFT_LIGHT, key: 'slope2_down_left_light', label: '2x1 DL Light', kind: 'solid', hex: tokens.voxels.slope2_down_left_light.hex },
	{ type: SLOPE3_UP_RIGHT_LIGHT, key: 'slope3_up_right_light', label: '3x1 UR Light', kind: 'solid', hex: tokens.voxels.slope3_up_right_light.hex },
	{ type: SLOPE3_UP_RIGHT_MID, key: 'slope3_up_right_mid', label: '3x1 UR Mid', kind: 'solid', hex: tokens.voxels.slope3_up_right_mid.hex },
	{ type: SLOPE3_UP_RIGHT_HEAVY, key: 'slope3_up_right_heavy', label: '3x1 UR', kind: 'solid', hex: tokens.voxels.slope3_up_right_heavy.hex },
	{ type: SLOPE3_UP_LEFT_HEAVY, key: 'slope3_up_left_heavy', label: '3x1 UL', kind: 'solid', hex: tokens.voxels.slope3_up_left_heavy.hex },
	{ type: SLOPE3_UP_LEFT_MID, key: 'slope3_up_left_mid', label: '3x1 UL Mid', kind: 'solid', hex: tokens.voxels.slope3_up_left_mid.hex },
	{ type: SLOPE3_UP_LEFT_LIGHT, key: 'slope3_up_left_light', label: '3x1 UL Light', kind: 'solid', hex: tokens.voxels.slope3_up_left_light.hex },
	{ type: SLOPE3_DOWN_RIGHT_HEAVY, key: 'slope3_down_right_heavy', label: '3x1 DR', kind: 'solid', hex: tokens.voxels.slope3_down_right_heavy.hex },
	{ type: SLOPE3_DOWN_RIGHT_MID, key: 'slope3_down_right_mid', label: '3x1 DR Mid', kind: 'solid', hex: tokens.voxels.slope3_down_right_mid.hex },
	{ type: SLOPE3_DOWN_RIGHT_LIGHT, key: 'slope3_down_right_light', label: '3x1 DR Light', kind: 'solid', hex: tokens.voxels.slope3_down_right_light.hex },
	{ type: SLOPE3_DOWN_LEFT_LIGHT, key: 'slope3_down_left_light', label: '3x1 DL Light', kind: 'solid', hex: tokens.voxels.slope3_down_left_light.hex },
	{ type: SLOPE3_DOWN_LEFT_MID, key: 'slope3_down_left_mid', label: '3x1 DL Mid', kind: 'solid', hex: tokens.voxels.slope3_down_left_mid.hex },
	{ type: SLOPE3_DOWN_LEFT_HEAVY, key: 'slope3_down_left_heavy', label: '3x1 DL', kind: 'solid', hex: tokens.voxels.slope3_down_left_heavy.hex },
];

export const VOXEL_DEFINITION_BY_TYPE = Object.fromEntries(
	VOXEL_DEFINITIONS.map((definition) => [definition.type, definition]),
) as Record<VoxelType, VoxelDefinition>;

export const VOXEL_TYPE_IDS = new Set<number>(
	VOXEL_DEFINITIONS.map((definition) => definition.type),
);

export const CELL_VERTEX_OFFSETS: Record<VoxelType, [number, number][]> = {
	1: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	2: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	3: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	4: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	5: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	6: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	7: [
		[0, 0],
		[1, 0],
		[1, 1],
	],
	8: [
		[0, 0],
		[1, 0],
		[0, 1],
	],
	9: [
		[0, 0],
		[0, 1],
		[1, 1],
	],
	10: [
		[1, 0],
		[1, 1],
		[0, 1],
	],
	11: [
		[0, 0],
		[1, 0],
		[1, 0.5],
	],
	12: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 0.5],
	],
	13: [
		[0, 0],
		[1, 0],
		[1, 0.5],
		[0, 1],
	],
	14: [
		[0, 0],
		[1, 0],
		[0, 0.5],
	],
	15: [
		[0, 0.5],
		[1, 1],
		[0, 1],
	],
	16: [
		[0, 0],
		[1, 0.5],
		[1, 1],
		[0, 1],
	],
	17: [
		[0, 0.5],
		[1, 0],
		[1, 1],
		[0, 1],
	],
	18: [
		[0, 1],
		[1, 0.5],
		[1, 1],
	],
	19: [
		[0, 0],
		[1, 0],
		[1, 1 / 3],
	],
	20: [
		[0, 0],
		[1, 0],
		[1, 2 / 3],
		[0, 1 / 3],
	],
	21: [
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 2 / 3],
	],
	22: [
		[0, 0],
		[1, 0],
		[1, 2 / 3],
		[0, 1],
	],
	23: [
		[0, 0],
		[1, 0],
		[1, 1 / 3],
		[0, 2 / 3],
	],
	24: [
		[0, 0],
		[1, 0],
		[0, 1 / 3],
	],
	25: [
		[0, 0],
		[1, 1 / 3],
		[1, 1],
		[0, 1],
	],
	26: [
		[0, 1 / 3],
		[1, 2 / 3],
		[1, 1],
		[0, 1],
	],
	27: [
		[0, 2 / 3],
		[1, 1],
		[0, 1],
	],
	28: [
		[0, 1],
		[1, 2 / 3],
		[1, 1],
	],
	29: [
		[0, 2 / 3],
		[1, 1 / 3],
		[1, 1],
		[0, 1],
	],
	30: [
		[0, 1 / 3],
		[1, 0],
		[1, 1],
		[0, 1],
	],
};

export const SLOPE_ROTATE_CW_MAP: Record<VoxelType, VoxelType> = {
	1: RIGID,
	2: SOFT,
	3: H_ACT,
	4: V_ACT,
	5: FIXED,
	6: CONTRACTILE,
	7: SLOPE_UP_LEFT,
	8: SLOPE_DOWN_RIGHT,
	9: SLOPE_DOWN_LEFT,
	10: SLOPE_UP_RIGHT,
	11: SLOPE2_UP_RIGHT_LIGHT,
	12: SLOPE2_UP_RIGHT_HEAVY,
	13: SLOPE2_UP_LEFT_HEAVY,
	14: SLOPE2_UP_LEFT_LIGHT,
	15: SLOPE2_DOWN_RIGHT_LIGHT,
	16: SLOPE2_DOWN_RIGHT_HEAVY,
	17: SLOPE2_DOWN_LEFT_HEAVY,
	18: SLOPE2_DOWN_LEFT_LIGHT,
	19: SLOPE3_UP_RIGHT_LIGHT,
	20: SLOPE3_UP_RIGHT_MID,
	21: SLOPE3_UP_RIGHT_HEAVY,
	22: SLOPE3_UP_LEFT_HEAVY,
	23: SLOPE3_UP_LEFT_MID,
	24: SLOPE3_UP_LEFT_LIGHT,
	25: SLOPE3_DOWN_RIGHT_HEAVY,
	26: SLOPE3_DOWN_RIGHT_MID,
	27: SLOPE3_DOWN_RIGHT_LIGHT,
	28: SLOPE3_DOWN_LEFT_LIGHT,
	29: SLOPE3_DOWN_LEFT_MID,
	30: SLOPE3_DOWN_LEFT_HEAVY,
};

export const SLOPE_MIRROR_X_MAP: Record<VoxelType, VoxelType> = {
	1: RIGID,
	2: SOFT,
	3: H_ACT,
	4: V_ACT,
	5: FIXED,
	6: CONTRACTILE,
	7: SLOPE_UP_LEFT,
	8: SLOPE_UP_RIGHT,
	9: SLOPE_DOWN_LEFT,
	10: SLOPE_DOWN_RIGHT,
	11: SLOPE2_UP_LEFT_LIGHT,
	12: SLOPE2_UP_LEFT_HEAVY,
	13: SLOPE2_UP_RIGHT_HEAVY,
	14: SLOPE2_UP_RIGHT_LIGHT,
	15: SLOPE2_DOWN_LEFT_LIGHT,
	16: SLOPE2_DOWN_LEFT_HEAVY,
	17: SLOPE2_DOWN_RIGHT_HEAVY,
	18: SLOPE2_DOWN_RIGHT_LIGHT,
	19: SLOPE3_UP_LEFT_LIGHT,
	20: SLOPE3_UP_LEFT_MID,
	21: SLOPE3_UP_LEFT_HEAVY,
	22: SLOPE3_UP_RIGHT_HEAVY,
	23: SLOPE3_UP_RIGHT_MID,
	24: SLOPE3_UP_RIGHT_LIGHT,
	25: SLOPE3_DOWN_LEFT_HEAVY,
	26: SLOPE3_DOWN_LEFT_MID,
	27: SLOPE3_DOWN_LEFT_LIGHT,
	28: SLOPE3_DOWN_RIGHT_LIGHT,
	29: SLOPE3_DOWN_RIGHT_MID,
	30: SLOPE3_DOWN_RIGHT_HEAVY,
};

export const GRID_MAJOR_EVERY = Number(tokens.metrics.grid_major_every);
export const EDGE_WIDTH_PX = Number(tokens.metrics.edge_width_px);

export function normalizeThemeMode(theme: string | null | undefined): ThemeMode {
	return theme === 'light' ? 'light' : 'dark';
}

export function getDesignThemeTokens(theme: ThemeMode): DesignThemeTokens {
	return tokens.themes[normalizeThemeMode(theme)];
}
