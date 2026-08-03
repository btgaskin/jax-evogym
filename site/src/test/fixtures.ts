import { readFileSync } from 'node:fs';

import type { WorldJson } from '../components/designer/types';

export function readWorldFixture(name: string): WorldJson {
	return JSON.parse(
		readFileSync(new URL(`../../../tests/experimental_slope_fixtures/${name}`, import.meta.url), 'utf8'),
	) as WorldJson;
}
