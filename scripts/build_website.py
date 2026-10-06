"""Copy the static website and version its local assets for each deployment."""
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlsplit, urlunsplit


def main():
    source = Path(__file__).resolve().parents[1] / 'docs'
    destination = Path(sys.argv[1])
    revision = sys.argv[2]
    shutil.copytree(source, destination, dirs_exist_ok=True)

    def version_asset(match):
        attribute, url = match.groups()
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or Path(parts.path).suffix not in {
            '.css', '.js', '.json', '.png', '.svg', '.ico', '.woff', '.woff2'
        }:
            return match.group(0)
        versioned = urlunsplit(('', '', parts.path, f'v={revision}', parts.fragment))
        return f'{attribute}="{versioned}"'

    for path in destination.rglob('*.html'):
        html = path.read_text().replace('__WEBSITE_REVISION__', revision)
        html = re.sub(r'(src|href)="([^"]+)"', version_asset, html)
        path.write_text(html)
    print(f'Website {revision} -> {destination}')


if __name__ == '__main__':
    main()
