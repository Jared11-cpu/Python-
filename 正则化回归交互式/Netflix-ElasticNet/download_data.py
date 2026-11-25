"""从 Kaggle 公开下载入口取回原始数据，不需要账号或密钥。"""
from pathlib import Path
import io
import hashlib
import urllib.request
import zipfile

root = Path(__file__).resolve().parent / 'data'
root.mkdir(exist_ok=True)
url = 'https://www.kaggle.com/api/v1/datasets/download/shivamb/netflix-shows'
with urllib.request.urlopen(url, timeout=60) as response:
    with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        data = archive.read('netflix_titles.csv')
expected = 'df1f4ad2027a5a14c3a33932ef0d4054565ff88adf92b7f263601b70fdc6f3f3'
if hashlib.sha256(data).hexdigest() != expected:
    raise RuntimeError('Kaggle 文件已变化，与本次报告的数据快照不一致；未覆盖本地文件。')
(root / 'netflix_titles.csv').write_bytes(data)
print(f'已保存 {len(data):,} 字节到 {root / "netflix_titles.csv"}')
