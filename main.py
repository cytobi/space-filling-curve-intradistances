from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import logging

class Curve(ABC):
    def __init__(self, size) -> None:
        self.size = size
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def get_matrix(self):
        pass

    def draw(self):
        plt.imshow(self.get_matrix())
        plt.title(f"{self.name.capitalize()} curve of size {self.size}")
        plt.savefig(f"curves/{self.name}.png", dpi=300)
        plt.clf()

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
    
class Peano(Curve): # Peano curve
    def create(self, iterations):
        if iterations == 1:
            return np.array([[2, 3, 8], [1, 4, 7], [0, 5, 6]])
        else:
            m = self.create(iterations-1)
            mh = np.fliplr(m)
            mv = np.flipud(m)
            mvh = np.fliplr(mv)
            m = np.block([[m+2*m.size, mv+3*m.size, m+8*m.size],
                          [mh+1*m.size, mvh+4*m.size, mh+7*m.size],
                          [m+0*m.size, mv+5*m.size, m+6*m.size]])
            return m
        
    def get_matrix(self):
        return self.create(self.size)
    
class Gray(Curve): # Gray code curve
    def create(self, iterations):
        if iterations == 1:
            return np.array([[1, 2], [0, 3]])
        else:
            m = self.create(iterations-1)
            mr = np.rot90(m, 2)
            m = np.block([[mr+1*m.size, mr+2*m.size], [m+0*m.size, m+3*m.size]])
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
    max_size = 256

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using max size {max_size}")

    max_size_log2 = int(np.log2(max_size))
    max_size_log3 = int(np.emath.logn(3, max_size))

    logging.info("Calculating Lines plot")
    lines_data = [(i, calculate_average_distance(Lines(i).get_matrix())) for i in range(2, max_size+1)]
    plt.plot(*zip(*lines_data), label="Lines", marker=".")

    logging.info("Calculating Hilbert plot")
    hilbert_data = [(2**i, calculate_average_distance(Hilbert(i).get_matrix())) for i in range(1, max_size_log2+1)]
    plt.plot(*zip(*hilbert_data), label="Hilbert", marker=".")

    logging.info("Calculating Z plot")
    z_data = [(2**i, calculate_average_distance(Z(i).get_matrix())) for i in range(1, max_size_log2+1)]
    plt.plot(*zip(*z_data), label="Z", marker=".")

    logging.info("Calculating Peano plot")
    peano_data = [(3**i, calculate_average_distance(Peano(i).get_matrix())) for i in range(1, max_size_log3+1)]
    plt.plot(*zip(*peano_data), label="Peano", marker=".")

    logging.info("Calculating Gray plot")
    gray_data = [(2**i, calculate_average_distance(Gray(i).get_matrix())) for i in range(1, max_size_log2+1)]
    plt.plot(*zip(*gray_data), label="Gray", marker=".")

    logging.info("Plotting")
    plt.legend()
    plt.xlabel("Size")
    plt.ylabel("Average distance to neighbouring cells")
    plt.savefig("plot.png", dpi=300)
    plt.clf()

    logging.info("Drawing curves")
    Lines(16).draw()
    Hilbert(4).draw()
    Z(4).draw()
    Peano(3).draw()
    Gray(4).draw()