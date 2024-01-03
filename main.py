from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

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
    

def gaussian_kernel(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def calculate_direct_distances(m):
    logging.debug(f"Calculating direct distances for matrix of size {m.shape}")

    len = m.shape[0]
    distances = []

    # calculate distance to right neighbor
    for i in range(len-1):
        for j in range(len):
            distances.append(abs(m[i][j] - m[i+1][j]))

    # calculate distance to bottom neighbor
    for i in range(len):
        for j in range(len-1):
            distances.append(abs(m[i][j] - m[i][j+1]))

    return distances

def calculate_weighted_distances(m, kernel):
    logging.debug(f"Calculating weighted distances for matrix of size {m.shape} with kernel of size {kernel.shape}")

    len = m.shape[0]
    kernel_size = kernel.shape[0]
    distances = []

    kernel = kernel / np.sum(kernel) # normalize kernel
    kernel = kernel * kernel_size**2 # scale kernel so that average value in kernel is 1

    for i in range(len):
        for j in range(len):
            for k in range(kernel_size):
                for l in range(kernel_size):
                    x = i + k - kernel_size // 2
                    y = j + l - kernel_size // 2
                    if x >= 0 and x < len and y >= 0 and y < len:
                        distances.append(abs(m[i][j] - m[x][y]) * kernel[k][l])
    
    return distances


if __name__ == "__main__":
    max_size = 256
    gkern_param = 5, 1

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using max size {max_size}")

    max_size_log2 = int(np.log2(max_size))
    max_size_log3 = int(np.emath.logn(3, max_size))

    gkern = gaussian_kernel(*gkern_param)

    logging.info("Calculating Lines plot")
    logging.info("Calculating direct distances")
    lines_direct_distances = [(i, calculate_direct_distances(Lines(i).get_matrix())) for i in tqdm(range(2, max_size+1))]
    lines_average_data = [(i, np.average(direct_distances)) for i, direct_distances in lines_direct_distances]
    lines_median_data = [(i, np.median(direct_distances)) for i, direct_distances in lines_direct_distances]
    logging.info("Calculating gaussian weighted distances")
    lines_gaussian_data = [(i, np.average(calculate_weighted_distances(Lines(i).get_matrix(), gkern))) for i in tqdm(range(2, max_size+1))]

    logging.info("Calculating Hilbert plot")
    logging.info("Calculating direct distances")
    hilbert_direct_distances = [(i, calculate_direct_distances(Hilbert(i).get_matrix())) for i in tqdm(range(1, max_size_log2+1))]
    hilbert_average_data = [(2**i, np.average(direct_distances)) for i, direct_distances in hilbert_direct_distances]
    hilbert_median_data = [(2**i, np.median(direct_distances)) for i, direct_distances in hilbert_direct_distances]
    logging.info("Calculating gaussian weighted distances")
    hilbert_gaussian_data = [(2**i, np.average(calculate_weighted_distances(Hilbert(i).get_matrix(), gkern))) for i in tqdm(range(1, max_size_log2+1))]

    logging.info("Calculating Z plot")
    logging.info("Calculating direct distances")
    z_direct_distances = [(i, calculate_direct_distances(Z(i).get_matrix())) for i in tqdm(range(1, max_size_log2+1))]
    z_average_data = [(2**i, np.average(direct_distances)) for i, direct_distances in z_direct_distances]
    z_median_data = [(2**i, np.median(direct_distances)) for i, direct_distances in z_direct_distances]
    logging.info("Calculating gaussian weighted distances")
    z_gaussian_data = [(2**i, np.average(calculate_weighted_distances(Z(i).get_matrix(), gkern))) for i in tqdm(range(1, max_size_log2+1))]

    logging.info("Calculating Peano plot")
    logging.info("Calculating direct distances")
    peano_direct_distances = [(i, calculate_direct_distances(Peano(i).get_matrix())) for i in tqdm(range(1, max_size_log3+1))]
    peano_average_data = [(3**i, np.average(direct_distances)) for i, direct_distances in peano_direct_distances]
    peano_median_data = [(3**i, np.median(direct_distances)) for i, direct_distances in peano_direct_distances]
    logging.info("Calculating gaussian weighted distances")
    peano_gaussian_data = [(3**i, np.average(calculate_weighted_distances(Peano(i).get_matrix(), gkern))) for i in tqdm(range(1, max_size_log3+1))]

    logging.info("Calculating Gray plot")
    logging.info("Calculating direct distances")
    gray_direct_distances = [(i, calculate_direct_distances(Gray(i).get_matrix())) for i in tqdm(range(1, max_size_log2+1))]
    gray_average_data = [(2**i, np.average(direct_distances)) for i, direct_distances in gray_direct_distances]
    gray_median_data = [(2**i, np.median(direct_distances)) for i, direct_distances in gray_direct_distances]
    logging.info("Calculating gaussian weighted distances")
    gray_gaussian_data = [(2**i, np.average(calculate_weighted_distances(Gray(i).get_matrix(), gkern))) for i in tqdm(range(1, max_size_log2+1))]

    logging.info("Plotting average distance curves")
    plt.plot(*zip(*lines_average_data), label="Lines", marker=".")
    plt.plot(*zip(*hilbert_average_data), label="Hilbert", marker=".")
    plt.plot(*zip(*z_average_data), label="Z", marker=".")
    plt.plot(*zip(*peano_average_data), label="Peano", marker=".")
    plt.plot(*zip(*gray_average_data), label="Gray", marker=".")
    plt.legend()
    plt.xlabel("Size")
    plt.ylabel("Average distance to neighbouring cells")
    plt.savefig("plot_average.png", dpi=300)
    plt.clf()

    logging.info("Plotting median distance curves")
    plt.plot(*zip(*lines_median_data), label="Lines", marker=".")
    plt.plot(*zip(*hilbert_median_data), label="Hilbert", marker=".")
    plt.plot(*zip(*z_median_data), label="Z", marker=".")
    plt.plot(*zip(*peano_median_data), label="Peano", marker=".")
    plt.plot(*zip(*gray_median_data), label="Gray", marker=".")
    plt.legend()
    plt.xlabel("Size")
    plt.ylabel("Median distance to neighbouring cells")
    plt.savefig("plot_median.png", dpi=300)
    plt.clf()

    logging.info("Plotting gaussian distance curves")
    plt.plot(*zip(*lines_gaussian_data), label="Lines", marker=".")
    plt.plot(*zip(*hilbert_gaussian_data), label="Hilbert", marker=".")
    plt.plot(*zip(*z_gaussian_data), label="Z", marker=".")
    plt.plot(*zip(*peano_gaussian_data), label="Peano", marker=".")
    plt.plot(*zip(*gray_gaussian_data), label="Gray", marker=".")
    plt.legend()
    plt.xlabel("Size")
    plt.ylabel("Gaussian weighted distance to neighbouring cells")
    plt.savefig("plot_gaussian.png", dpi=300)
    plt.clf()

    logging.info("Drawing curves")
    Lines(16).draw()
    Hilbert(4).draw()
    Z(4).draw()
    Peano(3).draw()
    Gray(4).draw()