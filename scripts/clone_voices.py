"""Generate cloned speech from reference clips, for evaluating the detector.

Runs in .venv-tts, NOT the detector venv -- the TTS engines pin their own
torch builds and would replace the CUDA torch the demo depends on.

Each engine writes to data/eval/<engine>/, and scripts/evaluate.py then scores
every folder through the real detector. The point is not volume: it is
covering distinct vocoder families, because that is the axis a detector
generalises along or fails to.

Usage (from the project root):

    .venv-tts/Scripts/python.exe scripts/clone_voices.py \
        --refs data/refs --engines xtts_v2 f5_tts
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.sentences import LANGUAGE_OF, all_sentences  # noqa: E402

TARGET_SR = 16000
OUT_ROOT = Path("data/eval")


def _ensure_ffmpeg_on_path() -> None:
    """Expose the bundled ffmpeg as plain "ffmpeg.exe" on PATH.

    imageio-ffmpeg ships the binary under a versioned name, but pydub (which
    f5-tts uses for audio I/O) shells out to "ffmpeg" and only warns when it
    is absent -- then fails later, somewhere less obvious.
    """
    try:
        import imageio_ffmpeg
    except Exception:
        return

    try:
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return

    bin_dir = Path(__file__).resolve().parent.parent / "data" / "bin"
    target = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

    if not target.exists():
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, target)

    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _load_dotenv() -> None:
    """Read .env into the environment.

    The Coqui licence acceptance lives there rather than being hardcoded, so
    that accepting is a visible, dated line in a file someone can read --
    not a condition buried in source that a future reader would assume had
    always been true.
    """
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

# Both engines decode via libsndfile/librosa, so mp3, m4a and flac references
# all load. Running scripts/prepare_refs.py first is still worth it: it gives
# every engine the identical decoded signal instead of letting each one's
# decoder path introduce its own differences.
REF_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


def _save_16k_mono(wav: np.ndarray, sr: int, path: Path) -> None:
    """Write as 16 kHz mono, matching what the pipeline actually ingests.

    Scoring engines at their native 24/44 kHz would compare them at different
    bandwidths and confound the result with resampling artifacts.
    """
    import librosa
    import soundfile as sf

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), TARGET_SR)


# --- engines ------------------------------------------------------------


class XTTSv2:
    """Coqui XTTS-v2. GAN vocoder, zero-shot, multilingual including Hindi.

    Closest to what is actually used for voice fraud today, and the engine
    that makes the Indic claim testable.

    LICENCE: XTTS-v2 ships under the Coqui Public Model Licence, which is
    non-commercial. The library refuses to run until COQUI_TOS_AGREED=1 is set
    in the environment. That is a licence acceptance, so it is left to you --
    this script will not set it on your behalf.
    """

    name = "xtts_v2"
    supports_hindi = True

    def __init__(self) -> None:
        _load_dotenv()
        if os.environ.get("COQUI_TOS_AGREED") != "1":
            raise RuntimeError(
                "XTTS-v2 requires accepting the Coqui Public Model Licence "
                "(non-commercial). Read it at "
                "https://coqui.ai/cpml, then set COQUI_TOS_AGREED=1 to proceed."
            )
        from TTS.api import TTS

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    def synth(self, text: str, ref: Path, out: Path) -> None:
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.tts.tts_to_file(
                text=text,
                speaker_wav=str(ref),
                language=LANGUAGE_OF.get(text, "en"),
                file_path=tmp_path,
            )
            wav, sr = sf.read(tmp_path, dtype="float32")
            _save_16k_mono(wav, sr, out)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class F5TTS:
    """F5-TTS. Flow-matching acoustic model with a Vocos decoder.

    A different artifact family from XTTS's GAN vocoder, which is the whole
    reason it is in the set: it is the likelier of the two to slip past a
    detector trained mostly on GAN-vocoded spoofs.

    English-first. Devanagari lines are skipped rather than fed to a model
    that would mangle them -- mangled speech is trivially detectable and would
    flatter the results.
    """

    name = "f5_tts"
    supports_hindi = False

    def __init__(self) -> None:
        from f5_tts.api import F5TTS as _F5

        self.tts = _F5()

    def synth(self, text: str, ref: Path, out: Path) -> None:
        import tempfile

        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            # ref_text="" makes F5 transcribe the reference itself.
            self.tts.infer(
                ref_file=str(ref), ref_text="", gen_text=text, file_wave=tmp_path
            )
            wav, sr = sf.read(tmp_path, dtype="float32")
            _save_16k_mono(wav, sr, out)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


ENGINES = {"xtts_v2": XTTSv2, "f5_tts": F5TTS}


# --- driver -------------------------------------------------------------


def run_engine(key: str, refs: list[Path], sentences: list[str], limit: int) -> None:
    factory = ENGINES[key]

    print(f"\n=== {key} ===")
    try:
        engine = factory()
    except Exception as exc:
        # A missing or unaccepted engine is skipped loudly, not fatally: the
        # other engine's results are still worth having. The traceback is
        # printed because the message alone is often useless -- a bare
        # "WinError 127" says nothing about which library failed to load.
        print(f"SKIPPED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=6)
        return

    usable = [
        s for s in sentences
        if engine.supports_hindi or LANGUAGE_OF.get(s, "en") == "en"
    ][:limit]

    if len(usable) < len(sentences[:limit]):
        print(f"  ({len(sentences[:limit]) - len(usable)} non-English lines skipped)")

    made = 0
    for ref in refs:
        speaker = ref.stem
        for i, text in enumerate(usable):
            out = OUT_ROOT / key / f"{speaker}__{i:02d}.wav"
            if out.exists():
                print(f"  skip (exists) {out.name}")
                made += 1
                continue
            try:
                engine.synth(text, ref, out)
                print(f"  {out.name}")
                made += 1
            except Exception:
                print(f"  FAILED {out.name}")
                traceback.print_exc(limit=2)

    print(f"{key}: {made} clip(s) in {OUT_ROOT / key}")


def _check_interpreter() -> None:
    """Refuse to run outside the TTS venv.

    Running this with the system interpreter or the detector venv produces
    confusing import failures rather than an obvious "wrong python", so it
    is worth one explicit check.
    """
    if "venv-tts" in sys.executable.replace("\\", "/"):
        return
    sys.exit(
        "Wrong interpreter: " + sys.executable + """
Run this with the TTS venv, the only one with the engines:
  .venv-tts/Scripts/python.exe scripts/clone_voices.py --refs data/refs"""
    )


def main() -> None:
    _check_interpreter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path, required=True,
                        help="folder of reference clips, one per speaker")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES),
                        choices=list(ENGINES))
    parser.add_argument("--limit", type=int, default=6,
                        help="sentences per speaker per engine")
    parser.add_argument("--no-hindi", action="store_true")
    args = parser.parse_args()

    refs = sorted(
        p for p in args.refs.glob("*")
        if p.is_file() and p.suffix.lower() in REF_SUFFIXES
    )
    if not refs:
        sys.exit(f"no reference clips in {args.refs} "
                 f"(run scripts/inspect_clips.py first)")

    _ensure_ffmpeg_on_path()
    print(f"{len(refs)} reference clip(s): {[r.stem for r in refs]}")

    sentences = all_sentences(include_hindi=not args.no_hindi,
                              include_mixed=not args.no_hindi)

    for key in args.engines:
        run_engine(key, refs, sentences, args.limit)

    print("\nNext: score everything with the detector venv —")
    print("  .venv/Scripts/python.exe scripts/evaluate.py --real data/refs")


if __name__ == "__main__":
    main()
