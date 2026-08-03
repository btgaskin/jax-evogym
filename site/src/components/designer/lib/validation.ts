import {
	ACTUATOR_TYPES,
	MULTI_CELL_SLOPE_TYPES,
	SLOPE_2X1_MATE_OFFSET,
	SLOPE_2X1_MATE_TYPE,
	SLOPE_2X1_TYPES,
	SLOPE_3X1_NEIGHBOR_RULES,
	SLOPE_3X1_TYPES,
	SLOPE_TYPES,
	STATIC_TERRAIN_TYPES,
	VOXEL_TYPE_IDS,
} from '../constants';
import type { DocumentPayload, EditorObject, ValidationIssue } from '../types';

function isPointInBounds(document: DocumentPayload, point: { x: number; y: number }) {
	return (
		point.x >= 0 &&
		point.x < document.gridWidth &&
		point.y >= 0 &&
		point.y < document.gridHeight
	);
}

function isConnected(object: EditorObject) {
	const occupied = new Set(
		object.voxels.map((voxel) => `${voxel.x}:${voxel.y}`),
	);

	if (!occupied.size) {
		return false;
	}

	const [first] = occupied;
	const stack = [first];
	const visited = new Set<string>();

	while (stack.length) {
		const current = stack.pop();

		if (!current || visited.has(current)) {
			continue;
		}

		visited.add(current);

		const [x, y] = current.split(':').map((value) => Number(value));
		for (const [dx, dy] of [
			[-1, 0],
			[1, 0],
			[0, -1],
			[0, 1],
		]) {
			const candidate = `${x + dx}:${y + dy}`;
			if (occupied.has(candidate) && !visited.has(candidate)) {
				stack.push(candidate);
			}
		}
	}

	return visited.size === occupied.size;
}

export function validateDocument(document: DocumentPayload): ValidationIssue[] {
	const issues: ValidationIssue[] = [];
	const seenNames = new Set<string>();
	const occupancy = new Map<string, string>();
	let hasActuator = false;

	if (document.gridWidth <= 0 || document.gridHeight <= 0) {
		issues.push({
			severity: 'error',
			code: 'invalid_grid_size',
			message: 'Grid width and height must be positive integers.',
		});
	}

	for (const object of document.objects) {
		if (seenNames.has(object.name)) {
			issues.push({
				severity: 'error',
				code: 'duplicate_name',
				message: `Duplicate object name: ${object.name}`,
				objectId: object.id,
			});
		}

		seenNames.add(object.name);

		const activeVoxels = object.voxels;
		const unknownVoxel = activeVoxels.find((voxel) => !VOXEL_TYPE_IDS.has(voxel.type));
		if (unknownVoxel) {
			issues.push({
				severity: 'error',
				code: 'unknown_voxel_type',
				message: `Object '${object.name}' contains unknown voxel type ID ${unknownVoxel.type}. Remove that cell or import a world that uses voxel types 1 through 30.`,
				objectId: object.id,
			});
		}

		if (!activeVoxels.length) {
			issues.push({
				severity: 'error',
				code: 'empty_object',
				message: `Object '${object.name}' has no voxels.`,
				objectId: object.id,
			});
			continue;
		}

		if (!isConnected(object)) {
			issues.push({
				severity: 'error',
				code: 'disconnected_object',
				message: `Connect all voxels in object '${object.name}' with edge-adjacent cells, or split the disconnected parts into separate objects.`,
				objectId: object.id,
			});
		}

		const activeTypes = new Set(activeVoxels.map((voxel) => voxel.type));
		const hasSlope = [...activeTypes].some((type) => SLOPE_TYPES.has(type));
		const hasMultiCellSlope = [...activeTypes].some((type) => MULTI_CELL_SLOPE_TYPES.has(type));

		if (hasSlope && [...activeTypes].some((type) => !STATIC_TERRAIN_TYPES.has(type))) {
			issues.push({
				severity: 'error',
				code: 'slope_requires_static_object',
				message: `Object '${object.name}' contains slope cells and must contain only FIXED and slope terrain cells.`,
				objectId: object.id,
			});
		}

		if (hasMultiCellSlope) {
			issues.push({
				severity: 'warning',
				code: 'slope2_rotate_disabled',
				message: `Object '${object.name}' contains 2x1/3x1 slopes; rotation is disabled for this object.`,
				objectId: object.id,
			});
		}

		const voxelByLocal = new Map(activeVoxels.map((voxel) => [`${voxel.x}:${voxel.y}`, voxel]));
		for (const voxel of activeVoxels) {
			if (!SLOPE_2X1_TYPES.has(voxel.type)) {
				continue;
			}
			const mateType = SLOPE_2X1_MATE_TYPE[voxel.type];
			const mateOffset = SLOPE_2X1_MATE_OFFSET[voxel.type];
			if (typeof mateType !== 'number' || !mateOffset) {
				continue;
			}
			const mateX = voxel.x + mateOffset[0];
			const mateY = voxel.y + mateOffset[1];
			const mate = voxelByLocal.get(`${mateX}:${mateY}`);
			if (!mate || mate.type !== mateType) {
				issues.push({
					severity: 'error',
					code: 'slope2_invalid_pair',
					message: `Object '${object.name}' has an invalid 2x1 slope pair at local (${voxel.x}, ${voxel.y}).`,
					objectId: object.id,
				});
				break;
			}
		}

		for (const voxel of activeVoxels) {
			if (!SLOPE_3X1_TYPES.has(voxel.type)) {
				continue;
			}
			const neighborRules = SLOPE_3X1_NEIGHBOR_RULES[voxel.type] ?? [];
			let invalidTriplet = false;
			for (const [dx, dy, expectedType] of neighborRules) {
				const mateX = voxel.x + dx;
				const mateY = voxel.y + dy;
				const mate = voxelByLocal.get(`${mateX}:${mateY}`);
				if (!mate || mate.type !== expectedType) {
					invalidTriplet = true;
					break;
				}
			}
			if (invalidTriplet) {
				issues.push({
					severity: 'error',
					code: 'slope3_invalid_triplet',
					message: `Object '${object.name}' has an invalid 3x1 slope triplet at local (${voxel.x}, ${voxel.y}).`,
					objectId: object.id,
				});
				break;
			}
		}

		for (const voxel of activeVoxels) {
			const worldX = object.origin.x + voxel.x;
			const worldY = object.origin.y + voxel.y;

			if (!(worldX >= 0 && worldX < document.gridWidth && worldY >= 0 && worldY < document.gridHeight)) {
				issues.push({
					severity: 'error',
					code: 'out_of_bounds',
					message: `Move object '${object.name}' so its voxel at (${worldX}, ${worldY}) is inside the grid, or enlarge the grid.`,
					objectId: object.id,
				});
				continue;
			}

			const key = `${worldX}:${worldY}`;

			if (occupancy.has(key)) {
				issues.push({
					severity: 'error',
					code: 'overlap',
					message: `Move one object or erase a voxel so '${occupancy.get(key)}' and '${object.name}' no longer overlap at (${worldX}, ${worldY}).`,
					objectId: object.id,
				});
			} else {
				occupancy.set(key, object.name);
			}

			if (ACTUATOR_TYPES.has(voxel.type)) {
				hasActuator = true;
			}
		}
	}

	const metadata = document.metadata ?? {
		spawn: null,
		target: null,
		direction: null,
		mirrorEnabled: false,
	};
	if (metadata.spawn && !isPointInBounds(document, metadata.spawn)) {
		issues.push({
			severity: 'warning',
			code: 'spawn_out_of_bounds',
			message: `Spawn point (${metadata.spawn.x}, ${metadata.spawn.y}) is outside the document grid.`,
		});
	}
	if (metadata.target && !isPointInBounds(document, metadata.target)) {
		issues.push({
			severity: 'warning',
			code: 'target_out_of_bounds',
			message: `Target point (${metadata.target.x}, ${metadata.target.y}) is outside the document grid.`,
		});
	}

	if (!hasActuator) {
		issues.push({
			severity: 'warning',
			code: 'no_actuator',
			message: 'Add an H-Act, V-Act, or Contractile voxel if this world should contain a controllable robot.',
		});
	}

	return issues;
}
