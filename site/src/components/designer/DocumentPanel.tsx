import { useEffect, useState } from 'react';

import { CaretDown, CaretRight } from '@phosphor-icons/react';

import { MAX_GRID_DIM } from './constants';
import { useEditorDispatch, useEditorState } from './state/context';

export default function DocumentPanel() {
	const state = useEditorState();
	const dispatch = useEditorDispatch();
	const [collapsed, setCollapsed] = useState(true);
	const [width, setWidth] = useState(String(state.document.gridWidth));
	const [height, setHeight] = useState(String(state.document.gridHeight));

	useEffect(() => {
		setWidth(String(state.document.gridWidth));
	}, [state.document.gridWidth]);

	useEffect(() => {
		setHeight(String(state.document.gridHeight));
	}, [state.document.gridHeight]);

	const commitGridSize = () => {
		dispatch({
			type: 'SET_GRID_SIZE',
			width: Number(width) || state.document.gridWidth,
			height: Number(height) || state.document.gridHeight,
		});
	};

	return (
		<section className="designer-panel w-full min-w-0 overflow-x-hidden rounded-[1.5rem] p-4">
			<button
				type="button"
				className="flex w-full items-center gap-2 text-left"
				onClick={() => setCollapsed((value) => !value)}
			>
				{collapsed ? <CaretRight size={14} className="text-muted" /> : <CaretDown size={14} className="text-muted" />}
				<h2 className="text-sm font-semibold uppercase text-muted">Document</h2>
				<span className="ml-auto font-mono text-xs text-muted">
					{state.document.gridWidth} x {state.document.gridHeight}
				</span>
			</button>
			{!collapsed && (
				<div className="mt-4 grid gap-3">
					<label className="grid gap-2 text-sm font-medium text-ink">
						Grid width
						<input
							type="number"
							min={1}
							max={MAX_GRID_DIM}
							step={1}
							value={width}
							onBlur={commitGridSize}
							onChange={(event) => setWidth(event.target.value)}
							onKeyDown={(event) => {
								if (event.key === 'Enter') event.currentTarget.blur();
							}}
							className="rounded-2xl border border-border bg-surface px-4 py-3 text-ink outline-none transition focus:border-accent focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1"
						/>
					</label>
					<label className="grid gap-2 text-sm font-medium text-ink">
						Grid height
						<input
							type="number"
							min={1}
							max={MAX_GRID_DIM}
							step={1}
							value={height}
							onBlur={commitGridSize}
							onChange={(event) => setHeight(event.target.value)}
							onKeyDown={(event) => {
								if (event.key === 'Enter') event.currentTarget.blur();
							}}
							className="rounded-2xl border border-border bg-surface px-4 py-3 text-ink outline-none transition focus:border-accent focus-visible:ring-2 focus-visible:ring-accent/40 focus-visible:ring-offset-1"
						/>
					</label>
				</div>
			)}
		</section>
	);
}
