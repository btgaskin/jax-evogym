declare global {
	interface Window {
		__jaxTheme?: {
			apply: (preference: 'auto' | 'light' | 'dark', emit?: boolean) => 'light' | 'dark';
			getPreference: () => 'auto' | 'light' | 'dark';
			getTheme: () => 'light' | 'dark';
			setPreference: (preference: 'auto' | 'light' | 'dark') => 'light' | 'dark';
		};
	}
}

export {};
