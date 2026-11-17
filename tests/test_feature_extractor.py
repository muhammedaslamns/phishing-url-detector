import pytest

from feature_extractor import extract_features, shannon_entropy


def test_returns_24_numeric_features():
    f = extract_features("example.com/index.html")
    assert len(f) == 24
    assert all(isinstance(v, (int, float)) for v in f.values())


def test_scheme_is_ignored():
    assert extract_features("http://example.com/a") == extract_features("example.com/a")
    assert extract_features("https://example.com/a") == extract_features("example.com/a")


def test_case_is_ignored():
    assert extract_features("EXAMPLE.com/A") == extract_features("example.com/a")


def test_url_length():
    assert extract_features("example.com")["url_length"] == 11


def test_ip_address_detected():
    f = extract_features("http://192.168.4.7/bank/login.php")
    assert f["has_ip_address"] == 1
    assert f["num_subdomains"] == 0


def test_normal_domain_is_not_ip():
    assert extract_features("google.com")["has_ip_address"] == 0


def test_at_symbol_detected():
    assert extract_features("paypal.com@evil.com/login")["has_at_symbol"] == 1
    assert extract_features("paypal.com/login")["has_at_symbol"] == 0


def test_double_slash_in_path():
    assert extract_features("site.com//redirect")["double_slash_in_path"] == 1
    assert extract_features("site.com/a/b")["double_slash_in_path"] == 0


def test_subdomain_count():
    assert extract_features("a.b.c.example.com/x")["num_subdomains"] == 3
    assert extract_features("example.com/x")["num_subdomains"] == 0


def test_url_shortener_detected():
    assert extract_features("bit.ly/abc123")["is_shortener"] == 1
    assert extract_features("google.com/search")["is_shortener"] == 0


def test_suspicious_words_counted():
    f = extract_features("192.168.4.7/bank/update-password/confirm.php")
    assert f["suspicious_word_count"] >= 3
    assert extract_features("example.com/photos")["suspicious_word_count"] == 0


def test_query_parameters_counted():
    assert extract_features("example.com/p?a=1&b=2")["num_query_params"] == 2
    assert extract_features("example.com/p")["num_query_params"] == 0


def test_entropy_values():
    assert shannon_entropy("") == 0
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("ab") == pytest.approx(1.0)