import { normalizeThemeMode, type ThemeMode } from './constants';
import { exportWorldFile, importWorldFile } from './lib/io';
import { STORAGE_KEY } from './state/reducer';
import type { DocumentPayload } from './types';

export interface BrowserDocumentHandle {
	fileName: string;
}

export interface BrowserOpenDocumentResult extends BrowserDocumentHandle {
	document: DocumentPayload;
}

export const browserStorageKey = STORAGE_KEY;

export function subscribeBrowserTheme(listener: (theme: ThemeMode) => void) {
	const syncTheme = () => {
		listener(normalizeThemeMode(document.documentElement.dataset.theme));
	};

	syncTheme();
	window.addEventListener('starlight-theme-change', syncTheme);

	const observer = new MutationObserver(syncTheme);
	observer.observe(document.documentElement, {
		attributeFilter: ['data-theme'],
		attributes: true,
	});

	return () => {
		window.removeEventListener('starlight-theme-change', syncTheme);
		observer.disconnect();
	};
}

export function getBrowserThemeMode() {
	return normalizeThemeMode(document.documentElement.dataset.theme);
}

export async function openBrowserDocument() {
	return await new Promise<BrowserOpenDocumentResult | null>((resolve, reject) => {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.json,application/json';
		input.hidden = true;

		const cleanup = () => input.remove();
		input.addEventListener(
			'change',
			async () => {
				const file = input.files?.[0];
				if (!file) {
					cleanup();
					resolve(null);
					return;
				}

				try {
					resolve({
						document: await importWorldFile(file),
						fileName: file.name,
					});
				} catch (error) {
					reject(error);
				} finally {
					cleanup();
				}
			},
			{ once: true },
		);

		document.body.append(input);
		input.click();
	});
}

export async function exportBrowserDocument(
	documentPayload: DocumentPayload,
	fileName = 'evogym-world.json',
) {
	exportWorldFile(documentPayload, fileName);
	return { fileName } satisfies BrowserDocumentHandle;
}

export function setBrowserPresentationState(
	state: { documentName: string | null; dirty: boolean },
) {
	const suffix = state.dirty ? ' *' : '';
	const name = state.documentName ?? 'Untitled';
	document.title = `${name}${suffix} | JAX EvoGym Designer`;
}
