import { useEffect, useRef } from 'react';
import type { Dispatch } from 'react';

import { validateDocument } from '../lib/validation';
import type { EditorAction, PersistentEditorState } from '../types';

export function useAutoValidation(state: PersistentEditorState, dispatch: Dispatch<EditorAction>) {
	const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	useEffect(() => {
		if (timerRef.current) {
			clearTimeout(timerRef.current);
		}

		timerRef.current = setTimeout(() => {
			dispatch({
				type: 'SET_ISSUES',
				issues: validateDocument(state.document),
			});
		}, 600);

		return () => {
			if (timerRef.current) {
				clearTimeout(timerRef.current);
			}
		};
	}, [state.document, dispatch]);
}
