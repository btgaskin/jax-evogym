import { createContext, useContext, useEffect, useReducer } from 'react';
import type { Dispatch, PropsWithChildren } from 'react';

import type { EditorAction, PersistentEditorState } from '../types';
import { createInitialEditorState, editorReducer, STORAGE_KEY } from './reducer';

const EditorStateContext = createContext<PersistentEditorState | null>(null);
const EditorDispatchContext = createContext<Dispatch<EditorAction> | null>(null);

export function EditorProvider({
	children,
	storageKey = STORAGE_KEY,
}: PropsWithChildren<{ storageKey?: string }>) {
	const [state, dispatch] = useReducer(editorReducer, storageKey, createInitialEditorState);

	useEffect(() => {
		try {
			localStorage.setItem(storageKey, JSON.stringify(state.document));
		} catch {
			// Ignore storage failures in restrictive browser contexts.
		}
	}, [state.document, storageKey]);

	return (
		<EditorStateContext.Provider value={state}>
			<EditorDispatchContext.Provider value={dispatch}>{children}</EditorDispatchContext.Provider>
		</EditorStateContext.Provider>
	);
}

export function useEditorState() {
	const state = useContext(EditorStateContext);
	if (!state) {
		throw new Error('useEditorState must be used within <EditorProvider>.');
	}
	return state;
}

export function useEditorDispatch() {
	const dispatch = useContext(EditorDispatchContext);
	if (!dispatch) {
		throw new Error('useEditorDispatch must be used within <EditorProvider>.');
	}
	return dispatch;
}

export function useEditor() {
	return {
		state: useEditorState(),
		dispatch: useEditorDispatch(),
	};
}
