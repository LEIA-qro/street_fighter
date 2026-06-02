import abc

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def train(self, env_fn, save_dir, steps):
        pass
        
    @abc.abstractmethod
    def resume(self):
        pass
        
    @abc.abstractmethod
    def tune(self):
        pass
        
    @abc.abstractmethod
    def test(self):
        pass
