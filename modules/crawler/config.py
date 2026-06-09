import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    'output_dir': str(PACKAGE_DIR / 'output'),
    'results_per_query': 10,
    'request_delay': 2.5,
    'request_timeout': 20,
    'max_retries': 3,
    'max_content_length': 80000,
    'min_content_length': 200,
    'target_encoding': 'utf-8',
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/111.0.0.0',
]

DEFAULT_QUERIES_FILE = str(PACKAGE_DIR / 'queries.txt')

QUERIES = None  # Will be loaded from file
