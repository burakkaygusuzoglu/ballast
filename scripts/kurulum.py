"""Kurulum ve doğrulama - 1_kurulum.bat tarafından sistem Python'u ile çalıştırılır.

Adımlar (her biri doğrulanır; ayrıntılar logs/kurulum.log dosyasına yazılır):
  1. Python sürümü ve yetki kontrolü
  2. Proje klasörü kontrolü (OneDrive, yol, yanlışlıkla oluşmuş .git klasörleri)
  3. Sanal ortam (.venv)
  4. Kütüphaneler (requirements.txt - sabit sürümler)
  5. Otomatik testler
  6. Binance bağlantı testi (API anahtarı gerekmez)
  7. Git deposu ve ilk kayıt (commit)

Yalnızca Python standart kütüphanesini kullanır (kütüphaneler henüz kurulmamış olabilir).
Tekrar çalıştırmak güvenlidir: var olan sanal ortam ve depo yeniden kullanılır.
"""
from __future__ import annotations

import contextlib
import os
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
IS_WINDOWS = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
LOG_PATH = ROOT / "logs" / "kurulum.log"
TOTAL_STEPS = 7
MIN_VERSION = (3, 11)
TESTED_VERSION = (3, 14)

FIRST_COMMIT = (
    "Kripto bot: ilk sürüm (kurulum doğrulandı)\n\n"
    "Strateji motoru, risk yönetimi, backtest, paper trading ve otomatik testler.\n"
)


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")  # noqa: SIM115 - log stays open for the whole run

    def write(self, text: str) -> None:
        if text:
            self.fh.write(text.rstrip("\n") + "\n")
            self.fh.flush()


LOG = Log(LOG_PATH)


def say(text: str = "") -> None:
    print(text, flush=True)
    LOG.write(text if text else " ")


def step(n: int, title: str) -> None:
    say()
    say(f"[{n}/{TOTAL_STEPS}] {title}")


def ok(text: str) -> None:
    say(f"      [OK]    {text}")


def warn(text: str) -> None:
    say(f"      [UYARI] {text}")


def fail(text: str) -> None:
    say(f"      [HATA]  {text}")


def ask(prompt: str) -> str:
    try:
        answer = input(prompt)
    except EOFError:
        answer = ""
    LOG.write(f"{prompt}{answer}")
    return answer.strip()


def run(cmd: list, timeout: int = 600, cwd: Path = ROOT, stdin_text: str | None = None) -> subprocess.CompletedProcess:
    """Komutu çalıştırır; tüm çıktıyı günlüğe yazar. Hiçbir durumda istisna fırlatmaz."""
    args = [str(c) for c in cmd]
    LOG.write(f"$ {' '.join(args)}")
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    try:
        cp = subprocess.run(args, check=False, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, env=env, input=stdin_text)
    except FileNotFoundError:
        cp = subprocess.CompletedProcess(args, 127, "", f"komut bulunamadı: {args[0]}")
    except subprocess.TimeoutExpired:
        cp = subprocess.CompletedProcess(args, 124, "", f"zaman aşımı ({timeout} sn)")
    LOG.write(cp.stdout)
    LOG.write(cp.stderr)
    LOG.write(f"(çıkış kodu {cp.returncode})")
    return cp


def show_tail(cp: subprocess.CompletedProcess, lines: int = 12) -> None:
    text = (cp.stdout or "") + (cp.stderr or "")
    for line in [ln for ln in text.splitlines() if ln.strip()][-lines:]:
        say(f"        | {line}")


def remove_tree(path: Path) -> None:
    def make_writable(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)   # git nesneleri Windows'ta salt okunurdur
        func(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable)
    else:  # pragma: no cover
        shutil.rmtree(path, onerror=make_writable)


# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------
def is_admin() -> bool:
    if not IS_WINDOWS:
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:
        return False


def check_python() -> bool:
    step(1, "Python kontrolü")
    version = sys.version_info
    if version < MIN_VERSION:
        fail(f"Python {platform.python_version()} çok eski; en az 3.11 gerekli (önerilen 3.14).")
        return False
    if sys.maxsize <= 2**32:
        fail("32-bit Python algılandı; 64-bit Python 3.14 kurun.")
        return False
    if is_admin():
        if os.environ.get("KRIPTO_BOT_ALLOW_ADMIN") == "1":   # yalnızca otomatik testler için
            warn("Yönetici yetkisi algılandı; KRIPTO_BOT_ALLOW_ADMIN=1 olduğu için devam ediliyor.")
        else:
            fail("Bu pencere YÖNETİCİ yetkisiyle açılmış. Kapatıp 1_kurulum.bat dosyasına normal çift "
                 "tıklayın (sağ tık > 'Yönetici olarak çalıştır' KULLANMAYIN).")
            return False
    ok(f"Python {platform.python_version()} (64-bit) - {sys.executable}")
    if (version.major, version.minor) != TESTED_VERSION:
        warn(f"Testler Python {TESTED_VERSION[0]}.{TESTED_VERSION[1]} ile yapıldı; sizdeki farklı. Devam ediliyor.")
    return True


# ---------------------------------------------------------------------------
# 2. Klasör
# ---------------------------------------------------------------------------
def stray_git_dirs() -> list[Path]:
    """Yanlış klasörde çalıştırılmış 'git init' sonucu oluşmuş olabilecek .git klasörleri."""
    home = Path.home()
    candidates = {home / ".git", home / "Desktop" / ".git", home / "OneDrive" / "Desktop" / ".git"}
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        if os.environ.get(name):
            candidates.add(Path(os.environ[name]) / "Desktop" / ".git")
    candidates.update(parent / ".git" for parent in ROOT.parents)
    return sorted(c for c in candidates if c.is_dir())


def repo_is_empty(git_dir: Path) -> bool | None:
    work = git_dir.parent
    commits = run(["git", "-c", "safe.directory=*", "-C", work, "rev-list", "-n", "1", "--all"])
    tracked = run(["git", "-c", "safe.directory=*", "-C", work, "ls-files"])
    if commits.returncode != 0 or tracked.returncode != 0:
        return None
    return not commits.stdout.strip() and not tracked.stdout.strip()


def check_folder() -> bool:
    step(2, "Proje klasörü kontrolü")
    path = str(ROOT)
    ok(f"Proje klasörü: {path}")
    if "onedrive" in path.lower():
        warn("Proje OneDrive ile eşitlenen bir klasörde; eşitleme kurulumu kilitleyebilir. Önerilen: C:\\yol\\canli-bot")
    if " " in path or not path.isascii():
        warn("Klasör yolunda boşluk ya da Türkçe/özel karakter var; bazı araçlar sorun çıkarabilir.")
    if not (ROOT / "requirements.txt").exists() or not (ROOT / "bot").is_dir():
        fail("Proje dosyaları eksik. Zip'in tamamının bu klasöre çıkarıldığından emin olun.")
        return False

    if shutil.which("git") is None:
        return True
    for git_dir in stray_git_dirs():
        empty = repo_is_empty(git_dir)
        if empty:
            warn(f"Boş bir git deposu bulundu: {git_dir}")
            say("              Muhtemelen bir komutun yanlış klasörde çalışmasıyla oluştu ve o klasördeki")
            say("              her şeyi bir git deposunun parçası gibi gösteriyor. İçinde hiç kayıt yok.")
            if ask("              Silinsin mi? (e/h): ").lower() in ("e", "evet", "y", "yes"):
                try:
                    remove_tree(git_dir)
                    ok(f"Silindi: {git_dir}")
                except OSError as exc:
                    warn(f"Silinemedi ({exc}); elle silebilirsiniz.")
            else:
                say("              Dokunulmadı.")
        elif empty is False:
            warn(f"Üst klasörde içinde kayıt olan bir git deposu var: {git_dir} (dokunulmadı).")
    return True


# ---------------------------------------------------------------------------
# 3. Sanal ortam
# ---------------------------------------------------------------------------
def ensure_venv() -> bool:
    step(3, "Sanal ortam (.venv)")
    wanted = f"{sys.version_info.major}.{sys.version_info.minor}"
    if VENV_PY.exists():
        cp = run([VENV_PY, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])
        if cp.returncode == 0 and cp.stdout.strip() == wanted:
            ok(f"Mevcut sanal ortam kullanılıyor (Python {wanted})")
            return True
        warn("Mevcut .venv farklı ya da bozuk bir Python'a ait; yeniden oluşturuluyor.")
        try:
            remove_tree(VENV)
        except OSError as exc:
            fail(f".venv silinemedi ({exc}). Açık pencereleri kapatıp tekrar deneyin.")
            return False
    cp = run([sys.executable, "-m", "venv", VENV], timeout=600)
    if cp.returncode != 0 or not VENV_PY.exists():
        fail("Sanal ortam oluşturulamadı.")
        show_tail(cp)
        return False
    ok(f"Sanal ortam oluşturuldu (Python {wanted})")
    return True


# ---------------------------------------------------------------------------
# 4. Kütüphaneler
# ---------------------------------------------------------------------------
def install_requirements() -> bool:
    step(4, "Kütüphaneler (sabit sürümler)")
    say("      İndirilip kuruluyor; ilk seferde 1-3 dakika sürebilir...")
    cp = run([VENV_PY, "-m", "pip", "install", "--only-binary=:all:", "--no-input", "-r", ROOT / "requirements.txt"],
             timeout=1800)
    if cp.returncode != 0:
        fail("Kütüphaneler kurulamadı. İnternet bağlantısını kontrol edin; sorun sürerse logs\\kurulum.log "
             "dosyasını inceleyin.")
        show_tail(cp)
        return False
    check = run([VENV_PY, "-m", "pip", "check"])
    if check.returncode != 0:
        fail("Kütüphaneler arasında uyumsuzluk var.")
        show_tail(check)
        return False
    count = len([ln for ln in run([VENV_PY, "-m", "pip", "list", "--format=freeze"]).stdout.splitlines() if "==" in ln])
    versions = run([VENV_PY, "-c", "import ccxt, pandas, numpy; "
                                   "print(f'ccxt {ccxt.__version__}, pandas {pandas.__version__}, numpy {numpy.__version__}')"])
    if versions.returncode != 0:
        fail("Kütüphaneler kuruldu ama içe aktarılamıyor.")
        show_tail(versions)
        return False
    ok(f"{count} paket kurulu ve uyumlu ({versions.stdout.strip()})")
    return True


# ---------------------------------------------------------------------------
# 5. Testler
# ---------------------------------------------------------------------------
def run_tests() -> bool:
    step(5, "Otomatik testler")
    say("      Çalışıyor (yaklaşık 30-90 sn)...")
    cp = run([VENV_PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], timeout=1800)
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
    summary = lines[-1].strip("= ") if lines else "çıktı yok"
    if cp.returncode == 0:
        ok(f"Tüm testler geçti: {summary}")
        return True
    fail(f"Testler başarısız: {summary}")
    for line in [ln for ln in lines if ln.startswith(("FAILED", "ERROR"))][:10]:
        say(f"        | {line}")
    return False


# ---------------------------------------------------------------------------
# 6. Binance
# ---------------------------------------------------------------------------
def check_binance() -> bool:
    step(6, "Binance bağlantısı (API anahtarı gerekmez)")
    cp = run([VENV_PY, ROOT / "scripts" / "baglanti_testi.py"], timeout=180)
    for line in [ln for ln in cp.stdout.splitlines() if ln.strip()]:
        say(f"      {line}")
    if cp.returncode != 0:
        fail("Binance'e bağlanılamadı. İnternet bağlantınızı ve güvenlik duvarını kontrol edin.")
        if cp.stderr.strip():
            show_tail(subprocess.CompletedProcess([], 1, "", cp.stderr), lines=6)
        return False
    return True


# ---------------------------------------------------------------------------
# 7. Git
# ---------------------------------------------------------------------------
def setup_git() -> bool:
    step(7, "Git deposu ve ilk kayıt")
    if shutil.which("git") is None:
        warn("Git bulunamadı; sürüm takibi atlandı (bot çalışmaya devam eder).")
        return True

    def git(*args: object, stdin_text: str | None = None) -> subprocess.CompletedProcess:
        return run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], stdin_text=stdin_text)

    if not (ROOT / ".git").exists():
        cp = git("init", "-b", "main")
        if cp.returncode != 0:
            fail("Git deposu oluşturulamadı.")
            show_tail(cp)
            return False
        ok("Git deposu oluşturuldu")
    else:
        ok("Mevcut git deposu kullanılıyor")
        plain = run(["git", "status", "--porcelain"])
        if "dubious ownership" in plain.stderr:
            warn("Git bu klasörün sahipliğini şüpheli buluyor (klasör yönetici yetkisiyle oluşturulmuş olabilir).")
            say("              Elle git kullanacaksanız bir kez şunu çalıştırın:")
            say(f"              git config --global --add safe.directory {ROOT.as_posix()}")

    if git("rev-parse", "--verify", "-q", "HEAD").returncode == 0:
        # Depoda zaten kayıt var: kullanıcının değişikliklerine dokunmayız, yalnızca bilgi veririz.
        changes = [ln for ln in git("status", "--porcelain").stdout.splitlines() if ln.strip()]
        if changes:
            ok(f"Depoda kayıtlar var; {len(changes)} dosyada kaydedilmemiş değişiklik (otomatik kayıt yapılmadı)")
        else:
            ok("Depo güncel; kaydedilmemiş değişiklik yok")
        return True

    name = git("config", "user.name").stdout.strip()
    email = git("config", "user.email").stdout.strip()
    if not name or not email:
        say("      Git kayıtları için bir ad ve e-posta gerekiyor (yalnızca bu projede saklanır).")
        name = name or ask("      Adınız: ")
        email = email or ask("      E-posta adresiniz: ")
        if not name or not email:
            warn("Ad/e-posta girilmedi; ilk kayıt atlandı. 1_kurulum.bat'ı tekrar çalıştırarak tamamlayabilirsiniz.")
            return True
        git("config", "user.name", name)
        git("config", "user.email", email)

    git("add", "-A")
    staged = git("diff", "--cached", "--name-only").stdout.splitlines()
    leaked = [f for f in staged if f == ".env" or f.startswith((".venv/", "data/", "logs/", "reports/"))]
    if leaked:
        git("reset", "-q")
        fail(f"Gizli/yerel dosyalar kayda girecekti, durduruldu: {', '.join(leaked[:5])}. .gitignore dosyasını kontrol edin.")
        return False
    if not staged:
        warn("Kaydedilecek dosya bulunamadı.")
        return True
    cp = git("commit", "-q", "-F", "-", stdin_text=FIRST_COMMIT)
    if cp.returncode != 0:
        fail("İlk kayıt (commit) oluşturulamadı.")
        show_tail(cp)
        return False
    short = git("rev-parse", "--short", "HEAD").stdout.strip()
    ok(f"İlk kayıt oluşturuldu ({short}, {len(staged)} dosya); .env, .venv ve veri klasörleri hariç")
    return True


# ---------------------------------------------------------------------------
def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(errors="replace")
    say("=" * 70)
    say(f" KRİPTO BOT - KURULUM ve DOĞRULAMA   ({datetime.now().astimezone():%Y-%m-%d %H:%M})")
    say("=" * 70)
    say(f" Sistem: {platform.system()} {platform.release()} | {platform.machine()}")

    results: dict[str, bool] = {}
    blocking = (("Python", check_python), ("Klasör", check_folder), ("Sanal ortam", ensure_venv),
                ("Kütüphaneler", install_requirements))
    for name, func in blocking:
        results[name] = func()
        if not results[name]:
            break
    if all(results.values()):
        results["Testler"] = run_tests()
        results["Binance"] = check_binance()
        results["Git"] = setup_git()

    say()
    say("=" * 70)
    if all(results.values()) and len(results) == TOTAL_STEPS:
        say(" KURULUM TAMAMLANDI - 7 adımın hepsi başarılı.")
        if (ROOT / "G1_golge_R1.bat").exists():            # v2: gölge botlar
            say(" Sıradaki adım: G1_golge_R1.bat ve G2_golge_R3.bat (gölge botlar, sanal para).")
        else:
            say(" Sıradaki adım: 2_veri_indir.bat (Binance'ten geçmiş veri, birkaç dakika sürer).")
        code = 0
    else:
        failed = [name for name, passed in results.items() if not passed]
        say(f" KURULUM TAMAMLANAMADI - sorunlu adım: {', '.join(failed)}")
        say(" Yukarıdaki [HATA] satırlarına bakın.")
        code = 1
    say(" Sorun olursa bu penceredeki çıktıyı inceleyin.")
    say(f" Ayrıntılı günlük: {LOG_PATH.relative_to(ROOT)}")
    say("=" * 70)
    return code


if __name__ == "__main__":
    sys.exit(main())
