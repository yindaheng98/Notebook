import re
import brotli
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
link = "https://zhuanlan.zhihu.com/p/348498294"
response = requests.get(link, headers={
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
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
content = str(soup.body.div.div.article.find_all("div", class_="Post-RichTextContainer")[0].div.div)
content = re.sub(r'<br/>', "\n", content)
content = re.sub(r'\n', "", content)

content = re.sub(r'<b>(.*?)</b>', lambda m:" **%s** " % m.group(1), content)
content = re.sub(r'<a[^>]*href="https://link.zhihu.com/\?target=([^">]+?)"[^>]*>(.*?)</a>', lambda m:"[%s](%s)" % (m.group(2), unquote(m.group(1))), content)
content = re.sub(r'<a[^>]*href="([^">]+?)"[^>]*>(.*?)</a>', lambda m:"[%s](%s)" % (m.group(2), m.group(1)), content)

content = re.sub(r'<source[^>]*/>', "", content)
content = re.sub(r'<picture><img[^>]*alt="([^"]+?)"[^>]*/></picture>', lambda m:"$$%s$$" % m.group(1), content)
content = re.sub(r'<img[^>]*alt="([^"]+?)"[^>]*/>', lambda m:"$%s$" % m.group(1), content)

def fig(m):
    return re.sub(r'<noscript><img[^>]*data-original="([^"]+?)"[^>]*/></noscript><img[^>]*>', lambda mm:"\n![](%s)\n" % mm.group(1), m.group(1))
content = re.sub(r'<figure[^>]*>(.*?)</figure>', fig, content)

def ol(m):
    return re.sub(r'<li[^>]*>(.*?)</li>', lambda mm:"1. %s\n" % mm.group(1), m.group(1))
content = re.sub(r'<ol>(.*?)</ol>', ol, content)

content = re.sub(r'<blockquote[^>]*>([^<]*?)</blockquote>', lambda m:"\n>%s\n" % m.group(1), content)
content = re.sub(r'<p[^>]*>([^<]*?)</p>', lambda m:"\n%s\n" % m.group(1), content)
content = re.sub(r'<h2[^>]*>([^<]*?)</h2>', lambda m:"\n## %s\n" % m.group(1), content)
content = re.sub(r'<h3[^>]*>([^<]*?)</h3>', lambda m:"\n### %s\n" % m.group(1), content)
content = re.sub(r'<h4[^>]*>([^<]*?)</h4>', lambda m:"\n#### %s\n" % m.group(1), content)
content = re.sub(r'<div[^>]*>', "", content)
content = re.sub(r'</div>', "", content)

content = re.sub(r'&gt;', ">", content)
content = re.sub(r'&lt;', "<", content)
content = re.sub(r'&amp;', "&", content)

with open("text.md", "w", encoding="utf8") as f:
    f.write(content)