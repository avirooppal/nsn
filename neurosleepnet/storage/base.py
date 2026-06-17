from abc import ABC, abstractmethod

class StorageAdapter(ABC):
    """
    Abstract interface for all storage adapters.
    """
    
    @abstractmethod
    def store(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    @abstractmethod
    def delete(self, *args, **kwargs):
        pass

    @abstractmethod
    def list(self, *args, **kwargs):
        pass

    @abstractmethod
    def search_keyword(self, *args, **kwargs):
        pass
