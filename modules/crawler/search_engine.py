import re
import time
import random
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from .config import USER_AGENTS, DEFAULT_CONFIG


class SearchEngine:
    def __init__(self, delay=None, timeout=None, max_retries=None, proxies=None):
        self.delay = delay or DEFAULT_CONFIG['request_delay']
        self.timeout = timeout or DEFAULT_CONFIG['request_timeout']
        self.max_retries = max_retries or DEFAULT_CONFIG['max_retries']
        self.proxies = proxies
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })

    def _rotate_ua(self):
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})

    def _rate_limit(self):
        jitter = random.uniform(0.5, 1.5)
        time.sleep(self.delay * jitter)

    def _fetch(self, url, referer=None):
        for attempt in range(self.max_retries):
            try:
                self._rotate_ua()
                headers = {}
                if referer:
                    headers['Referer'] = referer
                resp = self.session.get(
                    url, headers=headers, timeout=self.timeout,
                    proxies=self.proxies, allow_redirects=True,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return None

    def search(self, query, num_results=10, engines=('baidu', 'bing')):
        for engine in engines:
            self._rate_limit()
            try:
                method = getattr(self, f'_search_{engine}', None)
                if method:
                    results = method(query, num_results)
                    if results:
                        return results[:num_results]
            except Exception:
                continue
        return []

    def _parse_url_from_baidu_href(self, href):
        if not href or href.startswith('javascript'):
            return None
        if href.startswith('http://www.baidu.com/link?'):
            return href
        if href.startswith('/link?'):
            return 'http://www.baidu.com' + href
        if href.startswith('http'):
            return href
        return None

    def _search_baidu(self, query, num_results=10):
        encoded = quote_plus(query)
        url = f'https://www.baidu.com/s?wd={encoded}&rn={min(num_results * 2, 50)}'
        resp = self._fetch(url)
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, 'lxml')
        results = []
        seen_urls = set()

        for container in soup.select('.result, .c-container, .result-op, [class*="result"]'):
            title_el = container.select_one('h3 a, .t a, [class*="title"] a')
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get('href', '')
            url = self._parse_url_from_baidu_href(href)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            snippet_el = (
                container.select_one('.c-abstract, .content-right_8Zs40, '
                                     '.c-span-last, [class*="abstract"], '
                                     '[class*="summary"], .c-color-gray')
            )
            snippet = snippet_el.get_text(strip=True) if snippet_el else ''

            results.append({
                'title': title,
                'url': url,
                'snippet': snippet,
                'source': 'baidu',
            })

            if len(results) >= num_results:
                break

        if not results:
            return self._search_baidu_mobile(query, num_results)
        return results

    def _search_baidu_mobile(self, query, num_results=10):
        encoded = quote_plus(query)
        url = f'https://m.baidu.com/s?word={encoded}'
        resp = self._fetch(url, referer='https://m.baidu.com/')
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, 'lxml')
        results = []
        seen_urls = set()

        for item in soup.select('.result, .result-item, [class*="result"]'):
            title_el = item.select_one('a[href*="http"], .c-title a, [class*="title"] a')
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get('href', '')
            if not href or href in seen_urls or href.startswith('javascript'):
                continue
            seen_urls.add(href)

            snippet_el = item.select_one('.c-abstract, [class*="abstract"], .c-color-gray')
            snippet = snippet_el.get_text(strip=True) if snippet_el else ''

            results.append({
                'title': title,
                'url': href,
                'snippet': snippet,
                'source': 'baidu_mobile',
            })

            if len(results) >= num_results:
                break

        return results

    def _search_bing(self, query, num_results=10):
        encoded = quote_plus(query)
        url = f'https://www.bing.com/search?q={encoded}&count={num_results}'
        resp = self._fetch(url, referer='https://www.bing.com/')
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, 'lxml')
        results = []
        seen_urls = set()

        for item in soup.select('.b_algo'):
            title_el = item.select_one('h2 a')
            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            href = title_el.get('href', '')
            if not href or href in seen_urls:
                continue
            seen_urls.add(href)

            snippet_el = item.select_one('.b_caption p, .b_lineclamp2, .b_float')
            snippet = snippet_el.get_text(strip=True) if snippet_el else ''

            results.append({
                'title': title,
                'url': href,
                'snippet': snippet,
                'source': 'bing',
            })

            if len(results) >= num_results:
                break

        return results

    def resolve_url(self, url, timeout=None):
        to = timeout or self.timeout
        for attempt in range(self.max_retries):
            try:
                self._rotate_ua()
                resp = self.session.get(
                    url, timeout=to, proxies=self.proxies,
                    allow_redirects=True, stream=True,
                )
                resp.close()
                return resp.url
            except requests.RequestException:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return url
        return url
