import re
import time
import random

import requests
from bs4 import BeautifulSoup

from .config import USER_AGENTS, DEFAULT_CONFIG


CONTENT_SELECTORS = [
    'article',
    'main',
    '.post-content',
    '.article-content',
    '.article-detail',
    '.content-area',
    '#content',
    '.content',
    '.main-content',
    '.entry-content',
    '.post-body',
    '.article-body',
    '.detail-content',
    '.text-content',
    '.rich-content',
    '.topic-content',
    '.news-content',
    '.reader-content',
    '[class*="content"]',
    '[class*="article"]',
    '[class*="detail"]',
    '#article',
    '#main-content',
    '.md_content',
    '.doc-content',
    '.page-content',
]

REMOVE_SELECTORS = [
    'script', 'style', 'nav', 'header', 'footer', 'aside',
    '.advertisement', '.ad', '.ads', '.sidebar',
    '.comment', '.comments', '.comment-list',
    '.related-posts', '.recommend', '.recommend-list',
    '.share', '.share-box', '.social-share',
    '.copyright', '.footer', '.header',
    '.breadcrumb', '.breadcrumbs', '.nav',
    '.pagination', '.page-nav',
    '.subscribe', '.newsletter',
    'iframe', 'form', 'button',
]


class PageScraper:
    def __init__(self, timeout=None, max_content_length=None, proxies=None):
        self.timeout = timeout or DEFAULT_CONFIG['request_timeout']
        self.max_content_length = max_content_length or DEFAULT_CONFIG['max_content_length']
        self.proxies = proxies
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })

    def _rotate_ua(self):
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})

    def fetch(self, url):
        for attempt in range(DEFAULT_CONFIG['max_retries']):
            try:
                self._rotate_ua()
                resp = self.session.get(
                    url, timeout=self.timeout,
                    proxies=self.proxies, allow_redirects=True,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                if attempt < DEFAULT_CONFIG['max_retries'] - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return None

    def extract_main_html(self, html, url=''):
        soup = BeautifulSoup(html, 'lxml')

        for sel in REMOVE_SELECTORS:
            for el in soup.select(sel):
                el.decompose()

        title = ''
        title_el = soup.select_one(
            'h1, .post-title h1, .article-title h1, '
            '.entry-title, [class*="title"] h1, title'
        )
        if title_el:
            title = title_el.get_text(strip=True)

        main_content = None
        for selector in CONTENT_SELECTORS:
            el = soup.select_one(selector)
            if el and len(el.get_text(strip=True)) > 200:
                main_content = el
                break

        if not main_content:
            body = soup.find('body')
            if body:
                for tag in body.find_all(['p', 'div', 'section']):
                    text = tag.get_text(strip=True)
                    if len(text) > 100:
                        main_content = tag
                        break

        if not main_content:
            p_tags = soup.find_all('p')
            meaningful_ps = [p for p in p_tags if len(p.get_text(strip=True)) > 50]
            if meaningful_ps:
                wrapper = soup.new_tag('div')
                for p in meaningful_ps:
                    wrapper.append(p)
                main_content = wrapper

        if not main_content:
            main_content = soup.find('body') or soup

        html_out = str(main_content)
        if len(html_out) > self.max_content_length:
            html_out = html_out[:self.max_content_length]

        return title, html_out

    def scrape(self, url):
        try:
            resp = self.fetch(url)
            if not resp:
                return None, None

            content_type = resp.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return None, None

            try:
                raw = resp.content
                for enc in ('utf-8', 'gbk', 'gb2312', 'utf-16', 'iso-8859-1'):
                    try:
                        decoded = raw.decode(enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                else:
                    decoded = raw.decode('utf-8', errors='replace')
            except Exception:
                decoded = resp.text

            title, main_html = self.extract_main_html(decoded, url)
            if not main_html or len(main_html.strip()) < 50:
                return title, None

            return title, main_html

        except Exception:
            return None, None
