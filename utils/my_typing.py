import abc
from typing import TypeAlias, Sequence

IntSequence: TypeAlias = Sequence[int]
""" Superclass of `tuple[int]` and `list[int]`. (We often want to use both in one argument of function.) """
IntSequenceOrInt: TypeAlias = IntSequence | int
""" `IntSequence` possibly given by only one element -- `int`; see `seq_or_int_2_seq` """


def seq_or_int_2_seq(x: IntSequenceOrInt) -> IntSequence:
    """ Convert possible int to sequence by creating list containing only this int. """
    if isinstance(x, int):
        return [x]
    else:
        return x


def seq2str(x: Sequence) -> str:
    """ Unify string representation of `Sequence` """
    return str([elem for elem in x])


class String(abc.ABC):
    """
        Abstract class for supplying objects with some string
        (for example: if we want automatically create directory structure based on objects used).
    """
    # cannot be abc.abstractproperty, because tf.keras.Model.__init__(inputs=..., outputs=...) calls constructors
    # string = abc.abstractproperty
    string: str = None
    """ The string used for this class; see `String` """
