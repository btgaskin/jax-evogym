import { CheckCircle, WarningCircle } from '@phosphor-icons/react';

import type { SurfaceMessage } from './App';
import { cn } from './lib/cn';

interface ToastProps {
	message: SurfaceMessage | null;
}

export default function Toast({ message }: ToastProps) {
	return (
		<div className="pointer-events-none fixed inset-x-0 bottom-6 z-50 flex justify-center">
			<div
				className={cn(
					'pointer-events-auto inline-flex items-center gap-2 rounded-full border bg-surface px-4 py-2.5 text-sm shadow-lg transition-all duration-300 ease-out',
					message
						? 'translate-y-0 opacity-100'
						: 'pointer-events-none translate-y-4 opacity-0',
					message?.tone === 'error'
						? 'border-danger/20 text-danger'
						: 'border-success/20 text-success',
				)}
				aria-live="polite"
			>
				{message?.tone === 'error' ? (
					<WarningCircle size={16} weight="fill" />
				) : (
					<CheckCircle size={16} weight="fill" />
				)}
				<span>{message?.text}</span>
			</div>
		</div>
	);
}
