"""Check local links and fragments in an Astro build, without network access."""
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])


def check(root):
    root = root.resolve()
    pages = {p: Page(p.read_text()) for p in root.rglob('*.html')}
    errors = []
    count = 0
    for path, page in pages.items():
        route = '/' + path.relative_to(root).as_posix()
        if route.endswith('index.html'):
            route = route[:-10]
        for href in page.links:
            url = urlsplit(urljoin('https://jax-evogym.pages.dev' + route, href))
            if url.scheme not in ('https', 'http') or url.netloc != 'jax-evogym.pages.dev':
                continue
            target = root / unquote(url.path).lstrip('/')
            if target.is_dir():
                target /= 'index.html'
            elif not target.exists() and not target.suffix:
                target = target / 'index.html'
            count += 1
            if not target.is_file():
                errors.append(f'{route}: missing {href}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f'{route}: missing fragment {href}')
    print(f'Checked {count} local links across {len(pages)} HTML files.')
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)


if __name__ == '__main__':
    check(Path(sys.argv[1] if len(sys.argv) > 1 else 'site/dist'))
