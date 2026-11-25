from pathlib import Path
import pytest
from lob_alpha.real_data import discover_pairs, official_sample_url


def test_sample_url_is_official():
    url = official_sample_url('AAPL', 10)
    assert url.endswith('LOBSTER_SampleFile_AAPL_2012-06-21_10.zip')
    with pytest.raises(ValueError):
        official_sample_url('NOPE', 10)


def test_real_pair_discovery_sorted_and_checked(tmp_path):
    for day in ['2012-06-22', '2012-06-21']:
        (tmp_path / f'AAPL_{day}_34200000_57600000_message_10.csv').write_text('1,1,2,3,4,5')
        (tmp_path / f'AAPL_{day}_34200000_57600000_orderbook_10.csv').write_text('1,2,3,4')
    messages, books = discover_pairs(tmp_path, 'AAPL')
    assert '2012-06-21' in messages[0]
    assert len(messages) == len(books) == 2
    assert all('_orderbook_' in b for b in books)


def test_missing_pair_is_error(tmp_path):
    (tmp_path / 'AAPL_2012-06-21_34200000_57600000_message_10.csv').touch()
    with pytest.raises(FileNotFoundError):
        discover_pairs(tmp_path, 'AAPL')


def test_official_zip_extracts_expected_pairs_and_manifest(tmp_path, monkeypatch):
    from io import BytesIO
    from zipfile import ZipFile
    from lob_alpha import real_data
    archive = BytesIO()
    with ZipFile(archive, 'w') as z:
        for kind in ('message', 'orderbook'):
            name = f'LOBSTER_SampleFile_AAPL_2012-06-21_10/AAPL_2012-06-21_34200000_57600000_{kind}_10.csv'
            z.writestr(name, b'1,2,3,4\n')
    archive.seek(0)
    monkeypatch.setattr(real_data, 'urlopen', lambda req, timeout: BytesIO(archive.getvalue()))
    result = real_data.download_official_sample('AAPL', 10, tmp_path)
    assert result['symbol'] == 'AAPL'
    assert Path(result['files']['message']).read_bytes() == b'1,2,3,4\n'
    assert Path(result['files']['orderbook']).read_bytes() == b'1,2,3,4\n'
    assert (tmp_path / 'AAPL_2012-06-21_manifest.json').exists()
