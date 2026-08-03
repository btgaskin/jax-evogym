let selectIdCounter = 0;

export function createThemeSelectId(prefix: string): string {
	selectIdCounter += 1;
	return `${prefix}-${selectIdCounter.toString(36)}`;
}

export function __resetThemeSelectIdCounterForTests(): void {
	selectIdCounter = 0;
}
