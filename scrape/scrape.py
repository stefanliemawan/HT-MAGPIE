import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

REGEX_STRINGS = [
    r"article-content",
    r"c-blog-post__content",
    r"cbn-text-formatted",
    r"RichTextStoryBody",
    r"a-content",
    r"article-restofcontent",
    r"node__content",
    r"content-core",
    r"article__text",
    r"article__blocks",
    r"full-article",
    r"article-output",
    r"tds-content",
    r"td-post-content",
    r"entry-content",
    r"story[-]?text",
    r"Afg.*",
    r"ssrcss",
    r"^wsw$",
    r"story-two",
    r"et_pb_module et_pb_post_content et_pb_post_content_0_tb_body",  # new
    r"source-truthout",  # new, truthout
    r"article.*ody",
    r"mvp-content-main",
    r".*rticle_content",
    r"story-body.*",
    r"story_column",
    r".*-page-content",
    r"story-transcript",
    r"the_content_wrapper",
    r"body-description",
    r"body-text",
    r"body-content",
    r"post-body",
    r"<article>",
    r"^single-post$",
    r"^body$",
    r"^article$",
    r"^content$",
    r".*-content",
    r".*_content",
    r".*-article",
    r"article",
    r"content",
    r"body",
]

REGEX_COUNT = {key: 0 for key in REGEX_STRINGS}
REGEX_COUNT["<article>"] = 0
REGEX_COUNT["is this even used"] = 0


def parse_html(text):
    clean_text = re.sub(re.compile("<.*?>"), "", text)

    return clean_text


def get_soup(url):
    html_doc = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5
    ).content
    soup = BeautifulSoup(html_doc, "html.parser")

    return soup


def find_article(soup: BeautifulSoup, regex):
    article = None

    attrs = [{"class": regex}, {"id": regex}, {"itemprop": regex}]

    for attr in attrs:
        article = soup.find(attrs=attr)
        if not (article and len(article.get_text()) > 100):
            articles = soup.find_all(class_=regex)
            if articles:
                article = max(articles, key=lambda x: len(x.get_text()))

        if article and len(article.get_text()) > 100:
            break

    return article


def uniform_scrape(url):
    soup = get_soup(url)
    # print(soup)

    for regex_string in REGEX_STRINGS:
        # print(regex_string)
        if regex_string == r"<article>":
            article = soup.find("article")
            REGEX_COUNT["<article>"] += 1
        else:
            regex = re.compile(regex_string)
            article = find_article(soup, regex)

        if article and len(article.get_text()) > 100:
            REGEX_COUNT[regex_string] += 1
            break
        else:
            article = None

    # if not article:
    #     article = soup.find("article")
    #     REGEX_COUNT["<article>"] += 1

    content = ""
    for p in article.find_all("p"):
        parent_attrs = p.find_parent().attrs

        try:
            parent_class = parent_attrs["class"][0]
        except:
            parent_class = None

        if not (
            parent_class
            and (
                "meta" in parent_class
                or "promo" in parent_class
                or "share" in parent_class
                or "footer" in parent_class
                or "credit" in parent_class
                or "response_content" in parent_class
                or "author" in parent_class
                or "menu__link" in parent_class
                or "topper-hgroup" in parent_class
                or "tease" in parent_class
                or "donate" in parent_class
                or "comment" in parent_class
                or "newsletter" in parent_class
            )
        ):

            if p.find("strong"):
                p.strong.decompose()
            if p.find("em"):
                p.em.decompose()
            p_text = p.get_text(strip=True)
            content += p_text

    if (
        (not content or len(content) < 150)
        or url.startswith("https://beforeitsnews.com/")
        or url.startswith("https://www.billboard.com/")
    ):
        content = ""
        for p in article.find_all("p"):
            parent_attrs = p.find_parent().attrs

            try:
                parent_class = parent_attrs["class"][0]
            except:
                parent_class = None

            if not (
                parent_class
                and (
                    "meta" in parent_class
                    or "promo" in parent_class
                    or "share" in parent_class
                    or "footer" in parent_class
                    or "credit" in parent_class
                    or "response_content" in parent_class
                    or "author" in parent_class
                    or "menu__link" in parent_class
                    or "topper-hgroup" in parent_class
                    or "tease" in parent_class
                    or "donate" in parent_class
                    or "comment" in parent_class
                    or "newsletter" in parent_class
                )
            ):
                p_text = p.get_text(strip=True)
                content += p_text
        REGEX_COUNT["is this even used"] += 1

    print(content)
    print()
    print(REGEX_COUNT)

    return content


df = pd.read_csv("../dataset/articles_sorted_by_outlet_occurences.csv")


def scrape_content(row):
    url = row["article_url"]
    content = None

    try:
        print(f"\nscraping {row.name} {url}...\n")
        content = uniform_scrape(url)
    except KeyError:
        # print(f"KeyError for {url}")
        pass
    except Exception as exception:
        print(f"\n{url} failed, {exception}\n")

    return content


df["content"] = df.apply(scrape_content, axis=1)
print(f"{df["content"].count()} out of {len(df)} rows of articles text are scraped")
df.to_csv("scraped.csv", index=False)
