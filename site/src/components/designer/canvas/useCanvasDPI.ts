import { useEffect, useState } from 'react';
import type { RefObject } from 'react';

interface CanvasMetrics {
	width: number;
	height: number;
	dpr: number;
}

export function useCanvasDPI(
	containerRef: RefObject<HTMLDivElement | null>,
	canvasRef: RefObject<HTMLCanvasElement | null>,
) {
	const [metrics, setMetrics] = useState<CanvasMetrics>({
		width: 0,
		height: 0,
		dpr: 1,
	});

	useEffect(() => {
		const container = containerRef.current;
		const canvas = canvasRef.current;

		if (!container || !canvas) {
			return;
		}

		const update = () => {
			const rect = container.getBoundingClientRect();
			const width = Math.max(1, Math.round(rect.width));
			const height = Math.max(1, Math.round(rect.height));
			const dpr = window.devicePixelRatio || 1;

			if (
				canvas.width !== Math.round(width * dpr) ||
				canvas.height !== Math.round(height * dpr)
			) {
				canvas.width = Math.round(width * dpr);
				canvas.height = Math.round(height * dpr);
				canvas.style.width = `${width}px`;
				canvas.style.height = `${height}px`;
				const context = canvas.getContext('2d');
				context?.setTransform(dpr, 0, 0, dpr, 0, 0);
			}

			setMetrics((previous) => {
				if (previous.width === width && previous.height === height && previous.dpr === dpr) {
					return previous;
				}

				return {
					width,
					height,
					dpr,
				};
			});
		};

		const observer = new ResizeObserver(() => update());
		observer.observe(container);
		window.addEventListener('resize', update);
		window.visualViewport?.addEventListener('resize', update);
		update();

		return () => {
			observer.disconnect();
			window.removeEventListener('resize', update);
			window.visualViewport?.removeEventListener('resize', update);
		};
	}, [containerRef, canvasRef]);

	return metrics;
}
