import re

from .config import DEFAULT_CONFIG


class TextCleaner:
    def __init__(self, min_length=None):
        self.min_length = min_length or DEFAULT_CONFIG['min_content_length']

    def html_to_markdown(self, html):
        try:
            from markdownify import markdownify as md
            text = md(
                html,
                heading_style='ATX',
                bullets='-',
                strip=['img', 'figure', 'picture'],
            )
        except ImportError:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
            lines = []
            for el in soup.descendants:
                if el.name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    level = int(el.name[1])
                    prefix = '#' * level
                    text = el.get_text(strip=True)
                    if text:
                        lines.append(f'\n{prefix} {text}\n')
                elif el.name == 'p':
                    text = el.get_text(strip=True)
                    if text:
                        lines.append(f'{text}\n')
                elif el.name == 'li':
                    text = el.get_text(strip=True)
                    if text:
                        lines.append(f'- {text}')
                elif el.name == 'br':
                    lines.append('\n')
                elif el.name == 'hr':
                    lines.append('\n---\n')
                elif isinstance(el, str):
                    text = el.strip()
                    if text:
                        lines.append(text)
            text = '\n'.join(lines)
            return text

        text = self.remove_boilerplate(text)
        return text

    def remove_boilerplate(self, text):
        patterns = [
            r'(?i)copyright\s*[©(c)].*?(?:\n|$)',
            r'(?i)免责声明.*?(?:\n|$)',
            r'(?i)声明：.*?(?:\n|$)',
            r'(?i)如涉及侵权.*?(?:\n|$)',
            r'(?i)请联系我们.*?(?:\n|$)',
            r'(?i)点击.*?(?:阅读|查看|更多).*?(?:\n|$)',
            r'(?i)欢迎.*?(?:转载|分享).*?(?:\n|$)',
            r'(?i)未经.*?许可.*?(?:\n|$)',
            r'(?i)原文地址.*?(?:\n|$)',
            r'(?i)本文.*?(?:来源|出自|来自).*?(?:\n|$)',
            r'(?i)推荐阅读.*?(?:\n.*?){1,3}(?=\n|$)',
            r'(?i)相关(?:文章|阅读|推荐).*?(?:\n.*?){1,5}(?=\n|$)',
            r'\[.*?(?:广告|ad).*?\].*?(?:\n|$)',
            r'(?m)^\s*[-=*]{3,}\s*$',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return text

    def clean_text(self, text):
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)
        text = text.strip()
        return text

    def is_meaningful(self, text):
        if len(text) < self.min_length:
            return False
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        if total_chars == 0:
            return False
        ratio = chinese_chars / total_chars
        if ratio > 0.3 or chinese_chars > 100:
            return True
        english_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        if english_words > 20 and len(text) > 500:
            return True
        return False

    def process(self, html, title=''):
        md_text = self.html_to_markdown(html)
        md_text = self.remove_boilerplate(md_text)
        md_text = self.clean_text(md_text)
        return md_text
