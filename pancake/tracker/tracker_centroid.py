"""Tracker based on Centroid tracking."""
from .tracker import BaseTracker


def centroid(vertices):
    x_list = [vertex for vertex in vertices[::2]]
    y_list = [vertex for vertex in vertices[1::2]]
    x = sum(x_list) // len(x_list)
    y = sum(y_list) // len(y_list)
    return (x, y)


class CentroidTracker(BaseTracker):
    def __init__(self, *args, **kwargs) -> None:
        print("INIT CENTROID TRACKER")

    def update(self, det):  # det: list of koordinates x,y , x,y, ...
        centroids = [centroid(d) for d in det]
        return centroids
