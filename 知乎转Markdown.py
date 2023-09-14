import os
import re
import brotli
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from urllib.parse import unquote
link = "https://zhuanlan.zhihu.com/p/632637886"
response = requests.get(link, headers={
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,image/svg+xml,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "cache-control": "max-age=0",
    "referer": link,
    "sec-ch-ua": '" Not;A Brand";v="99", "Microsoft Edge";v="103", "Chromium";v="103"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.66 Safari/537.36 Edg/103.0.1264.44"
})
soup = BeautifulSoup(response.text, 'html.parser')
body = soup.body.div.div.article.find_all("div", class_="Post-RichTextContainer")[0].div.div
content = str(body)
content = re.sub(r'\n+', "\n", content)
content = re.sub(r' +', " ", content)
soup = BeautifulSoup(content, 'html.parser')

for div in soup('div'):
    if len(div.contents) != 1:
        continue
    for pre in div('pre'):
        if len(pre.contents) != 1:
            break
        for code in pre('code'):
            div.replace_with(f"\n```{code.attrs['class'][0]}\n{code.text}\n```\n")
for code in soup('code'):
    code.replace_with(f" `{code.text}` ")
for b in soup('b'):
    b.replace_with(f" **{b.text}** ")
for a in soup('a'):
    href = a.attrs['href']
    text = a.text
    if 'data-text' in a.attrs and len(text) < 1:
        text = a.attrs['data-text']
    m = re.match(r'https://link.zhihu.com/\?target=(.+?)$', href)
    if m is not None:
        href = unquote(m.group(1))
    a.replace_with(f"[{text}]({href})")


def get_url(url):
    os.makedirs("zhimg.com", exist_ok=True)
    r = requests.get(url)
    url = urlparse(url)
    with open("zhimg.com" + url.path, 'wb') as f:
        f.write(r.content)  # 写入二进制内容
    return "zhimg.com" + url.path


for noscript in soup.find_all("noscript"):
    noscript.extract()
for figure in soup('figure'):
    caption = ""
    for cap in figure('figcaption'):
        caption = cap.text
    for img in figure('img'):
        if len(caption) < 1 and 'data-caption' in img.attrs:
            caption = img.attrs['data-caption']
        if 'data-original' in img.attrs and len(img.attrs['data-original']) >= 1:
            image = img.attrs['data-original']
        else:
            image = img.attrs['data-actualsrc']
    figure.replace_with(f"\n![{caption}]({get_url(image)})\n")
for p in soup('p'):
    if len(p.contents) != 1:
        continue
    for tex in p('span'):
        if 'class' not in tex.attrs or "ztext-math" not in tex.attrs['class']:
            continue
        p.replace_with(f"\n$${tex.text}$$\n")
for tex in soup('span'):
    if 'class' not in tex.attrs or "ztext-math" not in tex.attrs['class']:
        continue
    tex.replace_with(f" ${tex.text}$ ")
for blockquote in soup('blockquote'):
    for br in soup('br'):
        br.replace_with('\n>')
    blockquote.replace_with(f"\n>{blockquote.text}\n")


def ul(el, prefix=""):
    for li in el('li'):
        for u in li('ul'):
            ul(u, prefix+'\t')
        li.replace_with(f"{prefix}* {li.text}\n")
    el.replace_with(f"\n{el.text}\n")

def ol(el, prefix=""):
    for li in el('li'):
        for u in li('ol'):
            ol(u, prefix+'\t')
        li.replace_with(f"{prefix}1. {li.text}\n")
    el.replace_with(f"\n{el.text}\n")


for u in soup('ul'):
    ul(u)
for o in soup('ol'):
    ol(o)

for p in soup('p'):
    p.replace_with(f"\n{p.text}\n")
for h1 in soup('h1'):
    h1.replace_with(f"\n# {h1.text}\n")
for h2 in soup('h2'):
    h2.replace_with(f"\n## {h2.text}\n")
for h3 in soup('h3'):
    h3.replace_with(f"\n### {h3.text}\n")
for h4 in soup('h4'):
    h4.replace_with(f"\n#### {h4.text}\n")
for h5 in soup('h5'):
    h5.replace_with(f"\n##### {h5.text}\n")

content = str(soup)
content = re.sub(r'&gt;', ">", content)
content = re.sub(r'&lt;', "<", content)
content = re.sub(r'&amp;', "&", content)

with open("text.md", "w", encoding="utf8") as f:
    f.write(content)
