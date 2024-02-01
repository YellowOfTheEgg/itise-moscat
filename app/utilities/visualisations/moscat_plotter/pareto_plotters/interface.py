from abc import ABC, abstractmethod


class IParetoPlotter(ABC):
    clevalot: object

    @abstractmethod
    def create_plot(self, img_path):
        pass
