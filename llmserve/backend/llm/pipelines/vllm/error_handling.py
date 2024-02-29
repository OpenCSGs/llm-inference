from abc import ABC, abstractmethod, abstractproperty

class ErrorReason(ABC):
    @abstractmethod
    def get_message(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_message()

    @abstractproperty
    def exception(self) -> Exception:
        raise NotImplementedError

    def raise_exception(self) -> Exception:
        raise self.exception

class ValidationError(ValueError):
    status_code = 400
    pass

class TooManyStoppingSequencesError(ValidationError):
    pass

class TooManyStoppingSequences(ErrorReason):
    def __init__(
        self, num_stopping_sequences: int, max_num_stopping_sequences: int
    ) -> None:
        self.num_stopping_sequences = num_stopping_sequences
        self.max_num_stopping_sequences = max_num_stopping_sequences

    def get_message(self) -> str:
        return (
            f"Too many stopping sequences. Recieved {self.num_stopping_sequences} stopping sequences,"
            f"but the maximum is {self.max_num_stopping_sequences}. Please reduce the number of provided stopping sequences."
        )

    @property
    def exception(self) -> Exception:
        return TooManyStoppingSequencesError(self.get_message())