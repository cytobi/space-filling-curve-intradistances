from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Curve(ABC):
    def __init__(self, size) -> None:
        self.size = size

    @abstractmethod
    def get_matrix(self):
        pass

class Lines(Curve): # horizontal line pattern from left to right and back
    def get_matrix(self):
        m = np.array([[i+j*self.size for i in range(self.size)] if j % 2 == 0 else [(self.size-i-1)+j*self.size for i in range(self.size)] for j in range(self.size)])
        return m
    
class Hilbert(Curve): # Hilbert curve
    def create(self, iterations):
        if iterations == 1:
            return np.array([[1, 2], [0, 3]])
        else:
            m = self.create(iterations-1)
            m = np.block([[m+1*m.size, m+2*m.size], [np.rot90(np.fliplr(m+0*m.size), 3), np.rot90(np.fliplr(m+3*m.size), 1)]])
            return m
        
    def get_matrix(self):
        return self.create(self.size)
    
class Z(Curve): # Z curve
    def create(self, iterations):
        if iterations == 1:
            return np.array([[0, 1], [2, 3]])
        else:
            m = self.create(iterations-1)
            m = np.block([[m+0*m.size, m+1*m.size], [m+2*m.size, m+3*m.size]])
            return m
        
    def get_matrix(self):
        return self.create(self.size)
    
        
def calculate_average_distance(m):
    len = m.shape[0]
    sum = 0

    # calculate distance to right neighbor
    for i in range(len-1):
        for j in range(len):
            sum += abs(m[i][j] - m[i+1][j])

    # calculate distance to bottom neighbor
    for i in range(len):
        for j in range(len-1):
            sum += abs(m[i][j] - m[i][j+1])

    return sum / (2 * (len-1) * len)


if __name__ == "__main__":
    max_size = 128

    lines_data = [(i, calculate_average_distance(Lines(i).get_matrix())) for i in range(2, max_size+1)]
    plt.plot(*zip(*lines_data), label="Lines", marker=".")

    max_size_log = int(np.log2(max_size))
    hilbert_data = [(2**i, calculate_average_distance(Hilbert(i).get_matrix())) for i in range(1, max_size_log+1)]
    plt.plot(*zip(*hilbert_data), label="Hilbert", marker=".")

    z_data = [(2**i, calculate_average_distance(Z(i).get_matrix())) for i in range(1, max_size_log+1)]
    plt.plot(*zip(*z_data), label="Z", marker=".")

    plt.legend()
    plt.xlabel("Size")
    plt.ylabel("Average distance to neighbouring cells")
    plt.savefig("plot.png", dpi=300)