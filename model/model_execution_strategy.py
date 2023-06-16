from abc import ABC, abstractmethod


class ModelExecutionStrategy(ABC):
    @abstractmethod
    def execute(self):
        pass
