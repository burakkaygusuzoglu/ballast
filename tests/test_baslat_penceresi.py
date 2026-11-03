"""Gerçek bot için güvenli başlatma penceresi: karar saatinden 1-30 dakika sonra."""
import baslat_penceresi as B
from bot.timeutil import parse_utc


def test_window_is_open_between_one_and_thirty_minutes_after_a_decision():
    assert B.window(parse_utc("2026-10-10T04:05")) == (True, parse_utc("2026-10-10T04:01"),
                                                       parse_utc("2026-10-10T04:30"))
    assert B.window(parse_utc("2026-10-10T04:01"))[0] and B.window(parse_utc("2026-10-10T04:30"))[0]


def test_window_points_to_the_next_decision_when_closed():
    late = B.window(parse_utc("2026-10-10T02:21"))                  # 00:00 kararından 2 saat sonra
    assert late == (False, parse_utc("2026-10-10T04:01"), parse_utc("2026-10-10T04:30"))
    assert B.window(parse_utc("2026-10-10T04:31"))[1] == parse_utc("2026-10-10T08:01")
    first_minute = B.window(parse_utc("2026-10-10T04:00:20"))       # veri henüz yayınlanmamış olabilir
    assert first_minute == (False, parse_utc("2026-10-10T04:01"), parse_utc("2026-10-10T04:30"))
    assert B.window(parse_utc("2026-10-10T23:59"))[1] == parse_utc("2026-10-11T00:01")    # gün dönümü


def test_main_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(B, "now_ms", lambda: parse_utc("2026-10-10T02:21"))
    assert B.main([]) == 1 and "[BEKLE]" in capsys.readouterr().out
    assert B.main(["--bilgi"]) == 0
    monkeypatch.setattr(B, "now_ms", lambda: parse_utc("2026-10-10T04:10"))
    assert B.main([]) == 0 and "[OK]" in capsys.readouterr().out
