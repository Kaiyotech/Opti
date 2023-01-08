from abc import ABC, abstractmethod
import numpy as np

class SelectionListener(ABC):
    @abstractmethod
    def on_selection(self, selected_model_name: str, model_action: np.ndarray):
        pass
