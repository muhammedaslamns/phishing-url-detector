import math
import re
from collections import Counter

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "paypal", "confirm", "webscr", "password", "ebay", "wallet", "billing",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc",
}

IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.\-]*://")


def shannon_entropy(text):
    """Measures how random the characters in a text look."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def split_url(url):
    """Split a URL into cleaned URL, hostname, path and query."""
    url = str(url).strip().lower()
    url = SCHEME_PATTERN.sub("", url)          # remove http:// or https://
    host = re.split(r"[/?#]", url, maxsplit=1)[0]
    rest = url[len(host):]
    host = host.split("@")[-1].split(":")[0]   # remove user info and port
    path = re.split(r"[?#]", rest, maxsplit=1)[0]
    query = rest.split("?", 1)[1] if "?" in rest else ""
    return url, host, path, query


def extract_features(url):
    """Turn one URL into a dictionary of numeric features."""
    url, host, path, query = split_url(url)

    has_ip = 1 if IP_PATTERN.match(host) else 0
    host_parts = host.split(".") if host else []
    num_subdomains = 0 if has_ip else max(len(host_parts) - 2, 0)
    tld = host_parts[-1] if (host_parts and not has_ip) else ""

    n = max(len(url), 1)
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)

    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "num_special_chars": sum(url.count(c) for c in "?=&%@#~"),
        "has_at_symbol": int("@" in url),
        "has_ip_address": has_ip,
        "num_subdomains": num_subdomains,
        "host_num_hyphens": host.count("-"),
        "host_num_digits": sum(c.isdigit() for c in host),
        "path_depth": path.count("/"),
        "num_query_params": (query.count("&") + 1) if query else 0,
        "suspicious_word_count": sum(url.count(w) for w in SUSPICIOUS_WORDS),
        "is_shortener": int(host in SHORTENERS),
        "tld_length": len(tld),
        "url_entropy": shannon_entropy(url),
        "double_slash_in_path": int("//" in path),
    }


if __name__ == "__main__":
    test = "http://paypal.secure-login.example.com/verify/account.php?id=123"
    for name, value in extract_features(test).items():
        print(f"{name}: {value}")