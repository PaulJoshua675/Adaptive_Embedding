from src.arxiv_loader import build_arxiv_pdf_url, normalize_arxiv_id


def test_normalize_arxiv_id_strips_prefix_and_version():
    assert normalize_arxiv_id("arXiv:2401.12345v2") == "2401.12345"


def test_build_arxiv_pdf_url_uses_pdf_endpoint():
    assert build_arxiv_pdf_url("2401.12345") == "https://arxiv.org/pdf/2401.12345.pdf"
