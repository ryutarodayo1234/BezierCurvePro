# 語彙の定義
characters = "abcdefghijklmnopqrstuvwxyz!'(),-.:;? "
# その他特殊記号
extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS>
]
_pad = "~"

# NOTE: パディングを 0 番目に配置
symbols = [_pad] + extra_symbols + list(characters)

additional_symbols = [
    'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1',
    'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
    'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
    'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
    'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
    'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',

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

symbols += additional_symbols

# 文字列⇔数値の相互変換のための辞書
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def num_vocab():
    """Get number of vocabraries

    Returns:
        int: Number of vocabraries

    Examples:

        >>> from ttslearn.tacotron.frontend.text import num_vocab
        >>> num_vocab()
        >>> 40
    """
    return len(symbols)


def text_to_sequence(text):
    """Convert text to sequence of numbers

    Args:
        text (str): Input text

    Returns:
        list: List of numbers

    Examples:

        >>> from ttslearn.tacotron.frontend.text import text_to_sequence
        >>> text_to_sequence("Hello world")
        >>> [1, 10, 7, 14, 14, 17, 39, 25, 17, 20, 14, 6, 2]

    """
    # 簡易のため、大文字と小文字を区別せず、全ての大文字を小文字に変換
    text = text.lower()

    # 文頭を表す<SOS>
    seq = [_symbol_to_id["^"]]

    # 本文
    seq += [_symbol_to_id[s] for s in text]

    # 文末を表す<EOS>
    seq.append(_symbol_to_id["$"])

    return seq


def sequence_to_text(seq):
    """Convert sequence of numbers to text

    Args:
        seq (list): Input sequence of numbers

    Returns:
        str: Text

    Examples:

        >>> from ttslearn.tacotron.frontend.text import sequence_to_text
        >>> sequence_to_text([1, 10, 7, 14, 14, 17, 39, 25, 17, 20, 14, 6, 2])
        >>> ['^', 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '$']

    """
    return [_id_to_symbol[s] for s in seq]
