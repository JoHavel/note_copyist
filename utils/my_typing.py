import abc
from typing import TypeAlias, Sequence

IntSequence: TypeAlias = Sequence[int]
IntSequenceOrInt: TypeAlias = IntSequence | int


def seq_or_int_2_seq(x: IntSequenceOrInt) -> IntSequence:
    if isinstance(x, int):
        return [x]
    else:
        return x


def seq2str(x: Sequence) -> str:
    return str([elem for elem in x])


class String(abc.ABC):
    # cannot be abc.abstractproperty, because tf.keras.Model.__init__(inputs=..., outputs=...) calls constructors
    # string = abc.abstractproperty
    string = None
