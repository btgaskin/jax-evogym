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
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Installation', slug: 'getting-started/installation' },
						{ label: 'Quick Start', slug: 'getting-started/quickstart' },
						{ label: 'Concepts', slug: 'getting-started/concepts' },
					],
				},
				{
					label: 'Guides',
					items: [
						{ label: 'Environments', slug: 'guides/environments' },
						{ label: 'Rendering', slug: 'guides/rendering' },
						{ label: 'Custom Environments', slug: 'guides/custom-environments' },
						{ label: 'Variable Morphology', slug: 'guides/variable-morphology' },
						{ label: 'Dense Grid Pipeline', slug: 'guides/dense-grid' },
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
					label: 'Core',
					items: [
						{ label: 'Architecture', slug: 'core/architecture' },
						{ label: 'Build API', slug: 'core/build-api' },
						{ label: 'Mirroring', slug: 'core/mirroring' },
						{ label: 'Parity', slug: 'core/parity' },
					],
				},
				{
					label: 'Designer',
					items: [
						{ label: 'User Guide', slug: 'designer/designer-guide' },
					],
				},
				{ label: 'Research & Data', slug: 'research' },
			],
		}),
	],
});
