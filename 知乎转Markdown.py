import os
import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from urllib.parse import unquote
# link = "https://zhuanlan.zhihu.com/p/707843145"
# response = requests.get(link, headers={
#     "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
#     "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
#     "cache-control": "max-age=0",
#     "priority": "u=0, i",
#     "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Microsoft Edge\";v=\"127\", \"Chromium\";v=\"127\"",
#     "sec-ch-ua-mobile": "?0",
#     "sec-ch-ua-platform": "\"Windows\"",
#     "sec-fetch-dest": "document",
#     "sec-fetch-mode": "navigate",
#     "sec-fetch-site": "cross-site",
#     "sec-fetch-user": "?1",
#     "upgrade-insecure-requests": "1",
#     "cookie": "__snaker__id=ooWNEq3vYqZ781tP; SESSIONID=LQiXtzpg5mzjfyDKYLWi5UpROakIuQJSUD9S91ISxg9; JOID=VV8UC02iWvpz0Fypca2rbKYuZ_Ng0y6hBZIlw0CQCLYyqRb6EguJRxrRWqZ2utxK5wvHymLh9d0JcRQWGP5ZBbw=; osd=VVoTCkuiX_1y1lysdqytbKMpZvVg1imgA5IgxEGWCLM1qBD6FwyIQRrUXadwutlN5g3Hz2Xg890MdhUQGPteBLo=; _xsrf=S8WqWd8BUUXPirZNTNq71bYUEXrIzk2C; _zap=215a42e9-d64a-4c1e-ab46-21a09438545d; d_c0=AFCQ7xX-uxiPTp1H402TBYa8ZDGW83Zkgx4=|1717712825; KLBRSID=dc02df4a8178e8c4dfd0a3c8cbd8c726|1720026847|1720026847; HMACCOUNT=B8AA57B9C4215583; __zse_ck=001_ROrF6cLFStobM0Zp7H3=/f=9DWlHh15ThNsl7VObWBpd8KOsS1gulhcWKA66IWVKOBCt7fHj60T6BfBXXOl86mc/ZwzMQ/QTElhUlsiHzrYut5NBkkarHUMY9aAbKHz9; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1723159234; z_c0=2|1:0|10:1723521959|4:z_c0|80:MS4xbEtWVEFRQUFBQUFtQUFBQVlBSlZUYWNwcUdkRjdyQmlDbTFaOVo4WEFpaFdwMXN3TXVWX1NRPT0=|48caf67ff61b822af6f5f8239387ba8b5d6bfb8e94144781af53c096501ecd92; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1723655752; BEC=36dafdc5edb6c00297b032c63dc4b447",
#     "Referer": link,
#     "Referrer-Policy": "unsafe-url"
# })

with open("text.html", "r", encoding='utf8') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

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
    if len([c for c in p.contents if c.strip is None or len(c.strip())>0]) != 1:
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
