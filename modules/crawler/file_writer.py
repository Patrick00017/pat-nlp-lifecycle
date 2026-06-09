import os
import re
from datetime import datetime
from pathlib import Path


def slugify(text, max_len=60):
    text = text.strip().lower()
    text = re.sub(r'[\s_]+', '_', text)
    text = re.sub(r'[^\w\u4e00-\u9fff\-]', '', text)
    text = re.sub(r'_+', '_', text)
    if len(text) > max_len:
        text = text[:max_len].rstrip('_')
    return text or 'untitled'


class FileWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.stats = {
            'total_queries': 0,
            'total_results': 0,
            'total_saved': 0,
            'total_skipped': 0,
            'by_query': {},
        }

    def save(self, title, content, query, url, index, source=''):
        query_slug = slugify(query, 40)
        query_dir = self.output_dir / query_slug
        query_dir.mkdir(parents=True, exist_ok=True)

        title_slug = slugify(title, 50) if title else f'result_{index:03d}'
        filename = f'{index:03d}_{title_slug}.md'
        filepath = query_dir / filename

        if filepath.exists():
            self.stats['total_skipped'] += 1
            return None

        metadata = (
            f'---\n'
            f'title: {title or "Untitled"}\n'
            f'source_url: {url}\n'
            f'search_query: {query}\n'
            f'source_engine: {source}\n'
            f'crawl_date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'---\n\n'
        )
        full_content = metadata + content

        filepath.write_text(full_content, encoding='utf-8')

        if query not in self.stats['by_query']:
            self.stats['by_query'][query] = {'found': 0, 'saved': 0}
        self.stats['by_query'][query]['saved'] += 1
        self.stats['total_saved'] += 1

        return str(filepath)

    def save_text(self, text, query, index, url=''):
        query_slug = slugify(query, 40)
        query_dir = self.output_dir / query_slug
        query_dir.mkdir(parents=True, exist_ok=True)

        filename = f'{index:03d}_text_{slugify(text[:30], 30)}.md'
        filepath = query_dir / filename

        metadata = (
            f'---\n'
            f'source_url: {url}\n'
            f'search_query: {query}\n'
            f'crawl_date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'---\n\n'
        )
        filepath.write_text(metadata + text, encoding='utf-8')

        if query not in self.stats['by_query']:
            self.stats['by_query'][query] = {'found': 0, 'saved': 0}
        self.stats['by_query'][query]['saved'] += 1
        self.stats['total_saved'] += 1

        return str(filepath)

    def print_stats(self):
        print(f'\n{"="*50}')
        print(f'CRAWL STATISTICS')
        print(f'{"="*50}')
        print(f'  Queries processed : {self.stats["total_queries"]}')
        print(f'  Results found     : {self.stats["total_results"]}')
        print(f'  Files saved       : {self.stats["total_saved"]}')
        print(f'  Files skipped     : {self.stats["total_skipped"]}')
        print(f'  Output directory  : {self.output_dir.resolve()}')
        print(f'{"-"*50}')
        for query, stats in self.stats['by_query'].items():
            print(f'  [{query[:50]}...]')
            print(f'    Found: {stats.get("found", 0)}, Saved: {stats["saved"]}')
        print(f'{"="*50}')
