#!/usr/bin/env python3
"""CLI crawler for corrugated paperboard manufacture industry text data.

Usage:
    python -m modules.crawler --help
    python -m modules.crawler --queries-file queries.txt
    python -m modules.crawler --queries "瓦楞纸板生产工艺" "corrugated board"
    python -m modules.crawler --engine baidu --results 5
"""

import argparse
import sys
import time
from pathlib import Path

from .config import DEFAULT_CONFIG, DEFAULT_QUERIES_FILE
from .search_engine import SearchEngine
from .page_scraper import PageScraper
from .text_cleaner import TextCleaner
from .file_writer import FileWriter


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        sanitized = [str(a).encode(
            sys.stdout.encoding or 'gbk', errors='replace'
        ).decode(sys.stdout.encoding or 'gbk', errors='replace') for a in args]
        print(*sanitized, **kwargs)


def load_queries(path):
    path = Path(path)
    if not path.exists():
        return []
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)
    return queries


def crawl(args):
    writer = FileWriter(args.output_dir)
    search = SearchEngine(
        delay=args.delay,
        timeout=args.timeout,
        max_retries=args.retries,
    )
    scraper = PageScraper(timeout=args.timeout)
    cleaner = TextCleaner()

    engines = args.engine
    if engines == ['auto']:
        engines = ('baidu', 'bing')
    else:
        engines = tuple(engines)

    queries = args.queries
    if args.queries_file:
        file_queries = load_queries(args.queries_file)
        if file_queries:
            queries = file_queries

    if not queries:
        safe_print('[ERROR] No search queries provided.')
        safe_print('  Use --queries-file or --queries to specify search terms.')
        sys.exit(1)

    writer.stats['total_queries'] = len(queries)
    safe_print(f'\nStarting crawl: {len(queries)} queries, {args.results} results/query\n')
    safe_print(f'  Engines       : {", ".join(engines)}')
    safe_print(f'  Output dir    : {args.output_dir}')
    safe_print(f'  Delay         : {args.delay}s')
    safe_print(f'  Max results   : {args.results}/query')
    safe_print(f'  Max pages     : {args.max_pages} total\n')
    safe_print('=' * 60)

    total_pages_fetched = 0
    start_time = time.time()

    for q_idx, query in enumerate(queries, 1):
        safe_print(f'\n[{q_idx}/{len(queries)}] Searching: {query}')
        writer.stats['by_query'][query] = {'found': 0, 'saved': 0}

        try:
            results = search.search(query, num_results=args.results)
        except Exception as e:
            safe_print(f'  [ERROR] Search failed: {e}')
            continue

        if not results:
            safe_print(f'  [SKIP] No results found')
            continue

        found = len(results)
        writer.stats['by_query'][query]['found'] = found
        writer.stats['total_results'] += found
        safe_print(f'  Found {found} results')

        for r_idx, result in enumerate(results, 1):
            if args.max_pages > 0 and total_pages_fetched >= args.max_pages:
                safe_print(f'\n  [STOP] Reached max page limit ({args.max_pages})')
                break

            url = result['url']
            title = result['title']
            snippet = result['snippet']
            source = result.get('source', 'unknown')

            safe_print(f'  [{r_idx}/{found}] Fetching: {title[:50] if title else url[:50]}...', end=' ')

            page_title, page_html = scraper.scrape(url)

            if not page_html:
                if snippet:
                    writer.save_text(
                        f"# {title}\n\nURL: {url}\n\n{snippet}",
                        query, r_idx, url,
                    )
                    safe_print('[SAVED snippet]')
                    writer.stats['by_query'][query]['saved'] += 1
                    writer.stats['total_saved'] += 1
                else:
                    safe_print('[SKIP]')
                total_pages_fetched += 1
                continue

            md_content = cleaner.process(page_html, title=page_title or title)

            if not cleaner.is_meaningful(md_content):
                if snippet:
                    writer.save_text(
                        f"# {title}\n\nURL: {url}\n\n{snippet}",
                        query, r_idx, url,
                    )
                    safe_print('[SAVED snippet]')
                    writer.stats['by_query'][query]['saved'] += 1
                    writer.stats['total_saved'] += 1
                else:
                    safe_print('[SKIP low content]')
                total_pages_fetched += 1
                continue

            char_count = len(md_content)
            final_title = page_title or title or 'Untitled'
            filepath = writer.save(final_title, md_content, query, url, r_idx, source)

            if filepath:
                safe_print(f'[OK {char_count} chars]')
            else:
                safe_print('[SKIP exists]')

            total_pages_fetched += 1

            if args.delay > 0:
                time.sleep(args.delay * 0.3)

        if args.max_pages > 0 and total_pages_fetched >= args.max_pages:
            break

    elapsed = time.time() - start_time
    safe_print(f'\n{"=" * 60}')
    safe_print(f'Crawl completed in {elapsed:.1f}s')
    writer.print_stats()

    return writer


def main():
    parser = argparse.ArgumentParser(
        description='Crawl corrugated paperboard manufacture industry text data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python -m modules.crawler\n'
            '  python -m modules.crawler --queries-file queries.txt --results 5\n'
            '  python -m modules.crawler --queries "瓦楞纸板" "corrugated board"\n'
            '  python -m modules.crawler --engine baidu --output ./my_data\n'
        ),
    )

    parser.add_argument(
        '--queries', nargs='+', default=None,
        help='Search queries (space-separated, use quotes for phrases)',
    )
    parser.add_argument(
        '--queries-file', default=None,
        help=f'File with queries (one per line, default: {DEFAULT_QUERIES_FILE})',
    )
    parser.add_argument(
        '-o', '--output-dir', default=DEFAULT_CONFIG['output_dir'],
        help=f'Output directory (default: {DEFAULT_CONFIG["output_dir"]})',
    )
    parser.add_argument(
        '-e', '--engine', nargs='+', default=['auto'],
        choices=['auto', 'baidu', 'bing'],
        help='Search engines to use (default: auto -> baidu, bing)',
    )
    parser.add_argument(
        '-n', '--results', type=int, default=DEFAULT_CONFIG['results_per_query'],
        help=f'Results per query (default: {DEFAULT_CONFIG["results_per_query"]})',
    )
    parser.add_argument(
        '--max-pages', type=int, default=0,
        help='Maximum total pages to fetch (0 = unlimited)',
    )
    parser.add_argument(
        '--delay', type=float, default=DEFAULT_CONFIG['request_delay'],
        help=f'Delay between requests in seconds (default: {DEFAULT_CONFIG["request_delay"]})',
    )
    parser.add_argument(
        '--timeout', type=int, default=DEFAULT_CONFIG['request_timeout'],
        help=f'Request timeout in seconds (default: {DEFAULT_CONFIG["request_timeout"]})',
    )
    parser.add_argument(
        '--retries', type=int, default=DEFAULT_CONFIG['max_retries'],
        help=f'Max retries per request (default: {DEFAULT_CONFIG["max_retries"]})',
    )

    args = parser.parse_args()

    if args.queries_file is None and args.queries is None:
        args.queries_file = DEFAULT_QUERIES_FILE

    crawl(args)


if __name__ == '__main__':
    main()
