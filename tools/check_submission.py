"""Audit the Git candidate set; optionally copy it to a fresh build/ directory.

Read-only by default. Never stages, commits, deletes, or overwrites an export.
Uses only the Python standard library and Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, help='Empty destination inside build/, e.g. build/submission_check/source')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    listing = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root)
    names = sorted(set(listing.decode('utf-8').strip('\0').split('\0')) - {''})
    candidates = {root / name for name in names}
    errors = []
    for name in names:
        if not (root/name).is_file():
            errors.append(f'Missing candidate: {name}')
        if name.startswith(('stage_records/', 'readme_materials/', 'build/')):
            errors.append(f'Local archive/build unexpectedly selected: {name}')

    def require(path, origin):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            errors.append(f'External dependency in {origin}: {path}')
        elif resolved not in candidates or not resolved.is_file():
            errors.append(f'Excluded or missing dependency in {origin}: {resolved.relative_to(root)}')

    link_count = 0
    documents = [root/'README.md', root/'external/THIRD_PARTY.md']
    documents += [p for p in candidates if p.parts[len(root.parts):len(root.parts)+1] == ('docs',) and p.suffix == '.md']
    for doc in documents:
        body = doc.read_text(encoding='utf-8-sig')
        for target in re.findall(r'\]\(([^)]+)\)', body):
            if re.match(r'\w+://|mailto:', target):
                continue
            location, _, fragment = unquote(target).partition('#')
            dest = (doc.parent/location).resolve() if location else doc
            require(dest, doc.relative_to(root))
            link_count += 1
            if fragment and dest.is_file() and dest.suffix == '.md':
                headings = re.findall(r'^#{1,6}\s+(.+)$', dest.read_text(encoding='utf-8-sig'), re.M)
                slugs = {re.sub(r'\s', '-', re.sub(r'[^\w\-\s]', '', h.lower())) for h in headings}
                if fragment not in slugs:
                    errors.append(f'Missing anchor in {doc.relative_to(root)}: {target}')

    scene_count = 0
    dependencies = set()
    for name in names:
        if name.startswith('scenes/') and name.endswith('.json'):
            scene_path = root/name
            scene = json.loads(scene_path.read_text(encoding='utf-8-sig'))
            if 'Camera' not in scene:
                continue
            scene_count += 1
            for obj in scene.get('Objects', []):
                if obj.get('TYPE') == 'mesh':
                    dependencies.add((scene_path.parent/obj['FILE']).resolve())
            for mat in scene.get('Materials', {}).values():
                if mat.get('BASE_COLOR_TEXTURE'):
                    dependencies.add((scene_path.parent/mat['BASE_COLOR_TEXTURE']).resolve())
    for path in list(dependencies):
        require(path, 'scene JSON')
        if path.is_file() and path.suffix == '.obj':
            for mtl in re.findall(r'^mtllib\s+(.+)$', path.read_text(encoding='utf-8'), re.M):
                require(path.parent/mtl.strip(), path.relative_to(root))
    # The renderer only loads texture images specified in JSON, not MTL map entries.
    cmake = (root/'CMakeLists.txt').read_text(encoding='utf-8')
    for filename in set(re.findall(r'(?:src|tests|tools)/[\w./-]+\.(?:cpp|cu|hpp|h)\b', cmake)):
        require(root/filename, 'CMakeLists.txt')
    for entry in json.loads((root/'docs/data/evidence_manifest.json').read_text(encoding='utf-8'))['files']:
        path = root/entry['file']
        require(path, 'evidence manifest')
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append(f'Changed evidence: {entry["file"]}')
    if errors:
        raise SystemExit('\n'.join(errors))

    summary = {'candidate_files': len(names), 'bytes': sum((root/n).stat().st_size for n in names),
               'markdown_links': link_count, 'scenes': scene_count,
               'scene_asset_files': len(dependencies), 'evidence_hashes': 'passed'}
    if args.export:
        dest = (root/args.export).resolve()
        build = (root/'build').resolve()
        if not dest.is_relative_to(build) or dest == build or (dest.exists() and any(dest.iterdir())):
            raise SystemExit('Export must be a new or empty directory strictly inside build/. No files were removed.')
        for name in names:
            target = dest/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, target)
        summary['export'] = dest.relative_to(root).as_posix()
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
