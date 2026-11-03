"""Komut satırı araçları uçtan uca: veri indir -> backtest (geliştirme + sınav) -> paper trading.

Borsa yerine sahte istemci kullanılır; geri kalan her şey (ayar dosyası, CSV'ler, raporlar,
grafik, SQLite durumu) gerçek dosyalarla, kullanıcının çalıştıracağı aynı kodla çalışır.
"""
import json

import pytest
import yaml

import backtest
import download_data
import run
from bot.config import CONFIG_ENV_VAR
from bot.data import candle_path
from helpers import H1, M15, TEST_CONFIG_PATH, FakeClient, make_candles, resample

N_DAYS = 150
EXAM_START = "2023-04-15"


@pytest.fixture(scope="module")
def market_data():
    data = {}
    for i, symbol in enumerate(["BTC/USDT", "ETH/USDT"]):
        m15 = make_candles(N_DAYS * 96, seed=40 + i, base=100.0 / (i + 1))
        data[(symbol, "15m")] = m15
        data[(symbol, "1h")] = resample(m15, H1)
    return data


@pytest.fixture
def project(tmp_path, monkeypatch, market_data):
    """Geçici bir proje: kendi ayar dosyası, veri, log ve rapor klasörleri."""
    raw = yaml.safe_load(TEST_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"].update(dir=str(tmp_path / "data"), start="2023-01-01", exam_start=EXAM_START)
    raw["logging"]["dir"] = str(tmp_path / "logs")
    raw["backtest"].update(reports_dir=str(tmp_path / "reports"), open_chart=False)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))

    end = int(market_data[("BTC/USDT", "15m")]["ts"].iloc[-1]) + M15
    client = FakeClient(market_data, now_ms=end + 5_000)
    factory = lambda *args, **kwargs: client  # noqa: E731
    monkeypatch.setattr(download_data, "ExchangeClient", factory)
    monkeypatch.setattr(run, "ExchangeClient", factory)
    return tmp_path, client


def test_download_then_backtest_dev_and_exam(project, capsys, monkeypatch):
    tmp_path, client = project
    assert download_data.main([]) == 0
    out = capsys.readouterr().out
    assert "[OK] Veri hazır" in out and out.count("[OK]    BTC/USDT") >= 2
    assert candle_path(tmp_path / "data", "binance", "ETH/USDT", "1h").exists()
    assert (tmp_path / "data" / "markets.json").exists()

    calls = client.source.calls
    assert download_data.main([]) == 0                       # ikinci kez: yalnızca güncelleme
    assert client.source.calls - calls <= 4
    capsys.readouterr()

    assert backtest.main(["--no-open"]) == 0
    out = capsys.readouterr().out
    assert "BACKTEST RAPORU - GELİŞTİRME DÖNEMİ" in out and "KARAR KAPISI" in out and "MALİYETLER" in out
    (dev_dir,) = [p for p in (tmp_path / "reports").iterdir() if p.name.endswith("_dev")]
    for name in ("rapor.txt", "ozet.json", "islemler.csv", "ozsermaye.csv", "grafik.png"):
        assert (dev_dir / name).stat().st_size > 0, name
    summary = json.loads((dev_dir / "ozet.json").read_text(encoding="utf-8"))
    assert summary["period"] == "dev" and summary["end"].startswith(EXAM_START)

    monkeypatch.setattr("builtins.input", lambda _prompt="": "hayir")
    assert backtest.main(["--period", "exam"]) == 1          # onay verilmedi -> sınav açılmaz
    assert not (tmp_path / "reports" / backtest.EXAM_LOG).exists()

    assert backtest.main(["--period", "exam", "--yes", "--no-plot"]) == 0
    log_lines = (tmp_path / "reports" / backtest.EXAM_LOG).read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1 and "config_sha256=" in log_lines[0]
    assert "SINAV DÖNEMİ" in capsys.readouterr().out


def test_paper_cli_once_status_and_reset(project, capsys, monkeypatch):
    tmp_path, client = project
    end = client.now
    client.now = end - 6 * H1                               # 6 saat önce başlatıldı
    assert run.main(["--once"]) == 0
    client.now = end                                        # 6 saat sonra tekrar
    assert run.main(["--once"]) == 0
    out = capsys.readouterr().out
    assert "kaldığı yerden devam" in out

    assert run.main(["--status"]) == 0
    assert "Paper trading özeti" in capsys.readouterr().out

    def no_keys(testnet=False):
        raise run.ConfigError(".env içinde BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET boş.")
    monkeypatch.setattr(run, "load_api_keys", no_keys)          # gerçek .env'den bağımsız
    assert run.main(["--mode", "live"]) == 2                    # anahtar yoksa canlı mod başlamaz
    assert "BINANCE_TESTNET_API_KEY" in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda _prompt="": "sil")
    client.now = end + 5 * M15
    assert run.main(["--reset-state", "--once"]) == 0
    assert "sıfırlandı" in capsys.readouterr().out


def test_backtest_without_data_gives_clear_message(project, capsys):
    assert backtest.main(["--no-open"]) == 2
    assert "2_veri_indir.bat" in capsys.readouterr().out


def test_config_error_exits_cleanly(project, capsys):
    tmp_path, _ = project
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_file.read_text(encoding="utf-8").replace("risk_per_trade", "risk_per_trad"),
                           encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        backtest.main([])
    assert exc.value.code == 2
    assert "risk_per_trad" in capsys.readouterr().out


def test_quick_edit_is_disabled_on_windows_console(monkeypatch):
    """Windows'ta pencereye tıklamak (Hızlı Düzenleme) botu dondurmasın; diğer konsol ayarları korunur."""
    import ctypes
    import sys

    from bot import cli

    class FakeKernel32:
        def __init__(self, mode, is_console=True):
            self.mode, self.is_console, self.set_to = mode, is_console, None

        def GetStdHandle(self, which):
            return 7

        def GetConsoleMode(self, handle, ref):
            ref._obj.value = self.mode
            return self.is_console

        def SetConsoleMode(self, handle, mode):
            self.set_to = mode

    k = FakeKernel32(0x1F7)                       # Windows varsayılanı: Hızlı Düzenleme açık
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": k})(), raising=False)
    cli.disable_quick_edit()
    assert k.set_to == 0x1B7                      # yalnızca 0x40 kapandı, 0x80 (genişletilmiş) açık

    redirected = FakeKernel32(0x1F7, is_console=False)
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": redirected})(), raising=False)
    cli.disable_quick_edit()
    assert redirected.set_to is None              # konsol değilse dokunulmaz


def test_named_shadow_bot_keeps_its_own_state_and_log(project, capsys, monkeypatch):
    tmp_path, client = project
    config_file = tmp_path / "golge.yaml"
    raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    raw["paper"]["name"] = "golge_x"
    config_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_file))
    end = client.now
    client.now = end - 6 * H1
    assert run.main(["--once"]) == 0
    client.now = end
    assert run.main(["--once"]) == 0
    out = capsys.readouterr().out
    assert "GÖLGE BOT (GOLGE_X)" in out
    assert (tmp_path / "data" / "golge_x_state.db").exists() and not (tmp_path / "data" / "paper_state.db").exists()
    assert (tmp_path / "logs" / "golge_x.log").exists()
    assert run.main(["--status"]) == 0
    assert "Gölge bot (golge_x) özeti" in capsys.readouterr().out
