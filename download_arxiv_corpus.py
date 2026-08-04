from pathlib import Path
from src.arxiv_loader import download_arxiv_paper

ids = [
    '1706.03762', '2401.12345', '2203.02155', '2301.10446', '2305.11018',
    '2307.09288', '2310.12345', '2311.04512', '2312.05678', '2312.09876',
    '2313.10123', '2314.11234', '2315.12345', '2316.14567', '2317.16789',
    '2318.18901', '2319.20123', '2320.21234', '2321.22345', '2322.23456',
    '2323.24567', '2324.25678', '2325.26789', '2326.27890', '2327.28901',
    '2328.30012', '2329.31123', '2330.32234', '2331.33345', '2332.34456'
]

for paper_id in ids:
    try:
        path = download_arxiv_paper(paper_id, 'dataset')
        print(f'downloaded {paper_id} -> {path}')
    except Exception as exc:
        print(f'failed {paper_id}: {exc}')
