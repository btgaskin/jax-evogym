import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';

interface ThemeBlock {
	name: 'light' | 'dark';
	variables: Record<string, string>;
}

function parseHexColor(raw: string) {
	const value = raw.trim().toLowerCase();
	if (!/^#[0-9a-f]{6}$/.test(value)) {
		throw new Error(`Expected 6-digit hex color, got: ${raw}`);
	}

	const r = Number.parseInt(value.slice(1, 3), 16);
	const g = Number.parseInt(value.slice(3, 5), 16);
	const b = Number.parseInt(value.slice(5, 7), 16);
	return { r, g, b };
}

function toRelativeLuminance(raw: string) {
	const { r, g, b } = parseHexColor(raw);
	const channels = [r, g, b].map((channel) => {
		const srgb = channel / 255;
		if (srgb <= 0.03928) {
			return srgb / 12.92;
		}
		return ((srgb + 0.055) / 1.055) ** 2.4;
	});
	return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function contrastRatio(foreground: string, background: string) {
	const l1 = toRelativeLuminance(foreground);
	const l2 = toRelativeLuminance(background);
	const lighter = Math.max(l1, l2);
	const darker = Math.min(l1, l2);
	return (lighter + 0.05) / (darker + 0.05);
}

function parseThemeBlock(css: string, name: 'light' | 'dark'): ThemeBlock {
	const marker = `/* Theme tokens: ${name} */`;
	const markerIndex = css.indexOf(marker);
	if (markerIndex === -1) {
		throw new Error(`Missing marker: ${marker}`);
	}

	const blockStart = css.indexOf('{', markerIndex);
	if (blockStart === -1) {
		throw new Error(`Could not locate opening brace for ${name} block`);
	}

	let depth = 0;
	let blockEnd = -1;
	for (let index = blockStart; index < css.length; index += 1) {
		const character = css[index];
		if (character === '{') {
			depth += 1;
		}
		if (character === '}') {
			depth -= 1;
			if (depth === 0) {
				blockEnd = index;
				break;
			}
		}
	}

	if (blockEnd === -1) {
		throw new Error(`Could not locate closing brace for ${name} block`);
	}

	const body = css.slice(blockStart + 1, blockEnd);
	const variables: Record<string, string> = {};
	for (const [, varName, varValue] of body.matchAll(/--([a-z0-9-]+):\s*([^;]+);/g)) {
		variables[varName] = varValue.trim();
	}

	return { name, variables };
}

function getVar(block: ThemeBlock, varName: string) {
	const value = block.variables[varName];
	if (!value) {
		throw new Error(`Missing CSS variable --${varName} in ${block.name} block`);
	}
	return value;
}

describe('theme token contracts', () => {
	const css = readFileSync(new URL('../styles/site.css', import.meta.url), 'utf8');
	const dark = parseThemeBlock(css, 'dark');
	const light = parseThemeBlock(css, 'light');

	test('starlight grayscale ordering is monotonic by mode', () => {
		const darkGrayLuminance = [1, 2, 3, 4, 5, 6].map((level) =>
			toRelativeLuminance(getVar(dark, `sl-color-gray-${level}`)),
		);
		const lightGrayLuminance = [1, 2, 3, 4, 5, 6].map((level) =>
			toRelativeLuminance(getVar(light, `sl-color-gray-${level}`)),
		);

		for (let index = 0; index < darkGrayLuminance.length - 1; index += 1) {
			expect(darkGrayLuminance[index]).toBeGreaterThan(darkGrayLuminance[index + 1]);
		}
		for (let index = 0; index < lightGrayLuminance.length - 1; index += 1) {
			expect(lightGrayLuminance[index]).toBeLessThan(lightGrayLuminance[index + 1]);
		}
	});

	test('AA contrast thresholds are met for primary, muted, and interactive text', () => {
		for (const block of [dark, light]) {
			const appBg = getVar(block, 'app-bg');
			const appInk = getVar(block, 'app-ink');
			const appMuted = getVar(block, 'app-muted');
			const appAccent = getVar(block, 'app-accent');
			const appOnAccent = getVar(block, 'app-on-accent');
			const appDanger = getVar(block, 'app-danger');
			const appOnDanger = getVar(block, 'app-on-danger');
			const accentText = getVar(block, 'sl-color-text-invert');
			const accentBg = getVar(block, 'sl-color-bg-accent');
			const docsText = getVar(block, 'sl-color-text');
			const docsBg = getVar(block, 'sl-color-bg');
			const docsAccent = getVar(block, 'sl-color-text-accent');

			expect(contrastRatio(appInk, appBg)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(appMuted, appBg)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(appAccent, appBg)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(appOnAccent, appAccent)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(appOnDanger, appDanger)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(docsText, docsBg)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(docsAccent, docsBg)).toBeGreaterThanOrEqual(4.5);
			expect(contrastRatio(accentText, accentBg)).toBeGreaterThanOrEqual(4.5);
		}
	});
});
