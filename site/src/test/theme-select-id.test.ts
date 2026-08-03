import { readFileSync } from 'node:fs';

import { beforeEach, describe, expect, test } from 'vitest';

import {
	__resetThemeSelectIdCounterForTests,
	createThemeSelectId,
} from '../components/theme/selectId';

describe('theme select id contracts', () => {
	beforeEach(() => {
		__resetThemeSelectIdCounterForTests();
	});

	test('createThemeSelectId returns unique IDs across repeated calls', () => {
		const first = createThemeSelectId('starlight-theme-preference-select');
		const second = createThemeSelectId('starlight-theme-preference-select');
		const third = createThemeSelectId('theme-preference-select');

		expect(first).not.toBe(second);
		expect(first).toMatch(/^starlight-theme-preference-select-/);
		expect(second).toMatch(/^starlight-theme-preference-select-/);
		expect(third).toMatch(/^theme-preference-select-/);
	});

	test('theme select Astro components use dynamic id/for bindings', () => {
		const starlightSource = readFileSync(
			new URL('../components/starlight/ThemeSelect.astro', import.meta.url),
			'utf8',
		);
		const inlineSource = readFileSync(
			new URL('../components/theme/ThemeSelectInline.astro', import.meta.url),
			'utf8',
		);

		expect(starlightSource).toContain(
			"createThemeSelectId('starlight-theme-preference-select')",
		);
		expect(starlightSource).toContain('for={selectId}');
		expect(starlightSource).toContain('id={selectId}');
		expect(starlightSource).not.toContain('id="starlight-theme-preference-select"');

		expect(inlineSource).toContain("createThemeSelectId('theme-preference-select')");
		expect(inlineSource).toContain('for={selectId}');
		expect(inlineSource).toContain('id={selectId}');
		expect(inlineSource).not.toContain('id="theme-preference-select"');
	});
});
