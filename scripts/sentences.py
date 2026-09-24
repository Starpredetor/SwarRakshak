"""The script the clones will speak.

Content is deliberately in-domain: these are the sentences a voice-fraud
attempt on a bank or an enterprise actually contains. Two reasons. It makes
the demo audio self-explanatory without narration, and it keeps the linguistic
material close to what the system would meet in deployment rather than
newspaper prose.

The Hindi and code-mixed lines exercise the Indic claim. The L1 layer is
supposed to be language-agnostic because vocoder artifacts are acoustic rather
than linguistic -- these are how you find out whether that holds.
"""
from __future__ import annotations

ENGLISH = [
    "Hello, this is Rahul from the accounts team. I need you to approve the transfer today.",
    "Please confirm the payment of four lakh rupees to the vendor account before five o'clock.",
    "I am travelling right now, so I cannot come to the office to sign it myself.",
    "Can you read out the one time password that was just sent to your phone?",
    "This is urgent and the managing director has already approved it verbally.",
    "Do not discuss this with anyone in the finance department until it is completed.",
]

HINDI = [
    "नमस्ते, मैं बैंक से बोल रहा हूँ, आपका खाता सत्यापित करना है।",
    "कृपया यह भुगतान आज शाम तक मंज़ूर कर दीजिए।",
    "मैं अभी यात्रा कर रहा हूँ, इसलिए मैं दफ़्तर नहीं आ सकता।",
]

CODE_MIXED = [
    "Sir, थोड़ा urgent है, please transfer approve कर दीजिए आज ही।",
    "मैंने अभी email भेजा है, आप एक बार check करके confirm कर दीजिए।",
    "Account details वही हैं, बस amount change हुआ है इस बार।",
]

# Neutral filler, useful when you want length without the fraud framing --
# for instance a clip you might share outside the team.
NEUTRAL = [
    "The weather has been unusually warm for this time of the year.",
    "I will send you the documents once I reach home this evening.",
]


def all_sentences(include_hindi: bool = True, include_mixed: bool = True) -> list[str]:
    out = list(ENGLISH)
    if include_hindi:
        out += HINDI
    if include_mixed:
        out += CODE_MIXED
    return out


# Language tags for engines that need one. XTTS takes a language code; F5-TTS
# is English-first, so the Devanagari lines are skipped for it rather than fed
# to a model that will mangle them and produce an unfairly easy detection.
LANGUAGE_OF: dict[str, str] = {
    **{s: "en" for s in ENGLISH + NEUTRAL},
    **{s: "hi" for s in HINDI},
    **{s: "hi" for s in CODE_MIXED},
}
