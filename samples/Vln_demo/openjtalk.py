import re

# 音素 (+pau/sil)
phonemes = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
    "sil",
]

extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
    "?",  # 文の末尾を表す特殊記号 <EOS> (疑問系)
    "_",  # ポーズ
    "#",  # アクセント句境界
    "[",  # ピッチの上がり位置
    "]",  # ピッチの下がり位置
]

_pad = "~"

# NOTE: 0 をパディングを表す数値とする
symbols = [_pad] + extra_symbols + phonemes

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

additional_symbols2 = [
    '_',

    'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1',
    'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
    'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
    'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
    'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
    'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',

    'D-1', 'E-1', 'G-1', 'A-1',
    'D-2', 'E-2', 'G-2', 'A-2',
    'D-3', 'E-3', 'G-3', 'A-3',
    'D-4', 'E-4', 'G-4', 'A-4',
    'D-5', 'E-5', 'G-5', 'A-5',
    'D-6', 'E-6', 'G-6', 'A-6',

    'whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th', '128th',

    'common', 'cut', '2/2', '2/4', '3/2', '3/4', '4/2', '4/4', '5/2', '5/4', '6/2', '6/4', '7/2', '7/4', '8/2', '8/4', '9/2', '9/4', '12/2', '12/4',

    'C major', 'G major', 'D major', 'A major', 'E major', 'B major', 'F# major', 'C# major',
    'F major', 'Bb major', 'Eb major', 'Ab major', 'Db major', 'Gb major', 'Cb major',
    'a minor', 'e minor', 'b minor', 'f# minor', 'c# minor', 'g# minor', 'd# minor', 'a# minor',
    'd minor', 'g minor', 'c minor', 'f minor', 'bb minor', 'eb minor', 'ab minor',
    'C lydian', 'G lydian', 'D lydian', 'A lydian', 'E lydian', 'B lydian', 'F# lydian', 'C# lydian',
    'C mixolydian', 'G mixolydian', 'D mixolydian', 'A mixolydian', 'E mixolydian', 'B mixolydian', 'F# mixolydian', 'C# mixolydian',
    'C dorian', 'G dorian', 'D dorian', 'A dorian', 'E dorian', 'B dorian', 'F# dorian', 'C# dorian',
    'C phrygian', 'G phrygian', 'D phrygian', 'A phrygian', 'E phrygian', 'B phrygian', 'F# phrygian', 'C# phrygian',
    'C aeolian', 'G aeolian', 'D aeolian', 'A aeolian', 'E aeolian', 'B aeolian', 'F# aeolian', 'C# aeolian',
    'C locrian', 'G locrian', 'D locrian', 'A locrian', 'E locrian', 'B locrian', 'F# locrian', 'C# locrian',

    'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'sffz',

    'grave', 'larghissimo', 'larghetto', 'largo', 'lento', 'adagio', 'andante', 'moderato', 'allegretto', 'allegro', 'vivace', 'presto', 'prestissimo',

    'staccato', 'tenuto', 'marcato', 'accent', 'portato', 'legato',

    'segno', 'coda', 'D.C.', 'D.S.', 'fine', 'To Coda', 'To Fine', 'To Segno', 'Da Capo', 'Dal Segno', 'Dal Segno al Coda', 'Da Capo al Fine', 'Da Capo al Segno', 'Da Capo al Coda',

    'fermata', 'fermata square', 'fermata triangle', 'fermata circle',

    'adagio', 'andante', 'moderato', 'allegretto', 'allegro', 'vivace', 'presto', 'prestissimo',
    'ritardando', 'rit.', 'accelerando', 'accel.', 'rallentando', 'rall.', 'stringendo', 'stretto', 'allargando', 'a tempo',

    'crescendo', 'cresc.', 'diminuendo', 'dim.', 'decrescendo', 'decresc.', 'crescendo diminuendo', 'crescendo-diminuendo', 'decrescendo crescendo', 'decrescendo-crescendo', 'hairpin', 'hairpin crescendo', 'hairpin diminuendo', 'hairpin decrescendo',

    '8va', '8vb', '15ma', '15mb', 'loco',

    'tremolo', 'trem.', 'tremolando', 'tremol.', 'tr.', 'trill',

    'accent', 'marcato', 'sforzando', 'sfz', 'strong accent', 'heavy accent', 'stress accent',

    'text', 'lyric', 'tempo', 'direction', 'annotation', 'rehearsal mark', 'dynamic mark', 'tempo mark', 'expression', 'articulation',
    'segno', 'coda', 'D.C.', 'D.S.', 'fine', 'To Coda', 'To Fine', 'To Segno', 'Da Capo', 'Dal Segno', 'Dal Segno al Coda', 'Da Capo al Fine', 'Da Capo al Segno', 'Da Capo al Coda',
]

def pp_symbols(labels, additional_symbols=None):
    # OpenJTalkラベルから韻律記号付き音素列を抽出する
    if additional_symbols:
        symbols = "|".join([r"\b" + re.escape(s) + r"\b" for s in additional_symbols])
        symbols += "|".join([r"\b" + re.escape(s) + r"\b" for s in additional_symbols2])
        symbols_pattern = f"({symbols})"
        symbols_pattern = re.compile(symbols_pattern)
    else:
        symbols_pattern = re.compile(r"\[.*?\]")
        
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)  # type: ignore

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)

        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")

    return PP


def num_vocab():
    """Get number of vocabraries

    Returns:
        int: Number of vocabraries

    Examples:

        >>> from ttslearn.tacotron.frontend.openjtalk import num_vocab
        >>> num_vocab()
        >>> 52
    """
    return len(symbols)


def text_to_sequence(text):
    """Convert phoneme + prosody symbols to sequence of numbers

    Args:
        text (list): text as a list of phoneme + prosody symbols

    Returns:
        list: List of numbers

    Examples:

        >>> from ttslearn.tacotron.frontend.openjtalk import text_to_sequence
        >>> text_to_sequence(["^", "m", "i", "[", "z","o", "$"])
        >>> [1, 31, 27, 6, 49, 35, 2]
    """
    return [_symbol_to_id[s] for s in text]


def sequence_to_text(seq):
    """Convert sequence of numbers to phoneme + prosody symbols

    Args:
        seq (list): Input sequence of numbers

    Returns:
        list: List of phoneme + prosody symbols

    Examples:

        >>> from ttslearn.tacotron.frontend.openjtalk import sequence_to_text
        >>> sequence_to_text([1, 31, 27, 6, 49, 35, 2])
        >>> ['^', 'm', 'i', '[', 'z', 'o', '$']
    """
    return [_id_to_symbol[s] for s in seq]
