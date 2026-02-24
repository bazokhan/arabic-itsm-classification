"""
Arabic text normalization for ITSM ticket classification.

Applies standard preprocessing steps before feeding text to MarBERTv2:
- Diacritic removal
- Alef normalization (ا أ إ آ → ا)
- Optional: teh marbuta normalization, latin lowercasing
"""

import re
import unicodedata


# Unicode ranges for Arabic diacritics (harakat)
_ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7\u06e8\u06ea-\u06ed]"
)

# Alef variants → bare alef
_ALEF_MAP = str.maketrans("أإآٱ", "اااا")

# Teh marbuta → heh
_TEH_MARBUTA_MAP = str.maketrans("ة", "ه")

# Whitespace normalization
_MULTI_SPACE = re.compile(r"\s+")


class ArabicTextNormalizer:
    """
    Lightweight Arabic normalizer for Egyptian ITSM text.

    MarBERTv2 was pretrained on raw Twitter text (including diacritics and
    variant alefs), so aggressive normalization is intentionally optional.
    The defaults here match what gives best results on informal Egyptian text.

    Parameters
    ----------
    remove_diacritics : bool
        Strip Arabic harakat. Default True.
    normalize_alef : bool
        Map أ إ آ ٱ → ا. Default True.
    normalize_teh_marbuta : bool
        Map ة → ه. Off by default; helps MSA models, may hurt dialect.
    lowercase_latin : bool
        Lowercase embedded Latin characters (e.g., "VPN" → "vpn"). Default True.
    """

    def __init__(
        self,
        remove_diacritics: bool = True,
        normalize_alef: bool = True,
        normalize_teh_marbuta: bool = False,
        lowercase_latin: bool = True,
    ):
        self.remove_diacritics = remove_diacritics
        self.normalize_alef = normalize_alef
        self.normalize_teh_marbuta = normalize_teh_marbuta
        self.lowercase_latin = lowercase_latin

    def normalize(self, text: str) -> str:
        """Normalize a single Arabic text string."""
        if not isinstance(text, str):
            return ""

        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        if self.remove_diacritics:
            text = _ARABIC_DIACRITICS.sub("", text)

        if self.normalize_alef:
            text = text.translate(_ALEF_MAP)

        if self.normalize_teh_marbuta:
            text = text.translate(_TEH_MARBUTA_MAP)

        if self.lowercase_latin:
            # Lower only ASCII latin chars, preserve Arabic
            text = re.sub(r"[A-Za-z]+", lambda m: m.group().lower(), text)

        # Collapse multiple spaces
        text = _MULTI_SPACE.sub(" ", text).strip()

        return text

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize_batch(self, texts: list[str]) -> list[str]:
        return [self.normalize(t) for t in texts]
