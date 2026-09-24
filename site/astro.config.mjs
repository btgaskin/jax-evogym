// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
	site: 'https://jax-evogym.pages.dev',
	vite: {
		plugins: [tailwindcss()],
	},
	integrations: [
		react(),
		starlight({
			title: 'JAX EvoGym',
			description: 'Core runtime docs and the browser-based voxel designer for soft-body robots.',
			customCss: ['/src/styles/site.css'],
			components: {
				SocialIcons: './src/components/starlight/HeaderLinks.astro',
				ThemeProvider: './src/components/starlight/ThemeProvider.astro',
				ThemeSelect: './src/components/starlight/ThemeSelect.astro',
			},
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/btgaskin/jax-evogym',
				},
			],
			// Group by reader task; order prerequisites before dependent workflows.
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Concepts', slug: 'getting-started/concepts' },
						{ label: 'Quick Start', slug: 'getting-started/quickstart' },
						{ label: 'First Simulation', slug: 'getting-started/first-simulation' },
					],
				},
				{
					label: 'Build and Control',
					items: [
						{ label: 'Designer Guide', slug: 'designer/designer-guide' },
						{ label: 'Designer to Simulation', slug: 'guides/designer-world' },
						{ label: 'Environments', slug: 'guides/environments' },
						{ label: 'Controllers and Actions', slug: 'guides/controllers' },
						{ label: 'Custom Environments', slug: 'guides/custom-environments' },
					],
				},
				{
					label: 'Evaluate and Evolve',
					items: [
						{ label: 'Evaluate Candidates', slug: 'guides/batched-evaluation' },
						{ label: 'Evolutionary Search', slug: 'guides/evolutionary-search' },
						{ label: 'Variable Morphology', slug: 'guides/variable-morphology' },
					],
				},
				{
					label: 'Supporting Guides',
					items: [
						{ label: 'Rendering', slug: 'guides/rendering' },
						{ label: 'Dense Grid Pipeline', slug: 'guides/dense-grid' },
						{ label: 'Troubleshooting', slug: 'guides/troubleshooting' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'API Boundary', slug: 'reference/api-boundary' },
						{ label: 'Voxel Types', slug: 'reference/voxel-types' },
						{ label: 'World JSON Format', slug: 'reference/world-json' },
						{ label: 'Types', slug: 'reference/types' },
						{ label: 'Observation Helpers', slug: 'reference/observation-helpers' },
						{ label: 'Simulation Loop', slug: 'reference/simulation-loop' },
					],
				},
				{
					label: 'Core Internals',
					items: [
						{ label: 'Architecture', slug: 'core/architecture' },
						{ label: 'Build API', slug: 'core/build-api' },
						{ label: 'Mirroring', slug: 'core/mirroring' },
						{ label: 'Parity', slug: 'core/parity' },
					],
				},
				{ label: 'Research & Data', slug: 'research' },
			],
		}),
	],
});
