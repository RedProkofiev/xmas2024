from multiprocessing import Pool, cpu_count

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


class BeerCabinetWithdrawal:
    """
    Highly sophisticated class that spews out N * f(B | BC) given a case N and K.
    Run it a couple hundred thousand/million times and believe in the central limit theorem -
    For any given N and K above 1, it should converge at a figure stated in the proof.
    """

    def __init__(self, N: int, K: int, seed: int = 42):
        self.rng = np.random.default_rng(seed=seed)
        self.N = N
        self.K = K
        self.H = []

        self.reset()

        self.max_run = [(self.BC, "", "", -np.inf)]
        self.min_run = [(self.BC, "", "", np.inf)]

    def reset(self) -> None:
        self.counter = 0
        self.LD = -1
        self.BC = np.repeat(np.arange(1, self.N + 1), self.K)
        self.H.clear()
        # syntax: BC, H, removed beer, counter
        self.run_data = [(self.BC, "", "", self.counter)]

    def __f(self) -> list[int]:
        # self.BC, in its current state.
        # self.H is empty.
        H_seq = np.random.permutation(np.arange(len(self.BC)))
        for idx in H_seq:
            self.H.append(self.BC[idx])
            if (self.H.count(self.BC[idx]) > 1) and (self.BC[idx] != self.LD):
                self.counter += len(
                    self.H
                )  # add to the counter whatever is in your hands
                self.LD = self.BC[idx]  # note whatever you drank
                new_BC = np.delete(
                    self.BC, idx
                )  # and throw away the bottle (into recycling <3)
                h_copy = np.copy(self.H)
                self.H.clear()
                self.run_data.append((new_BC, h_copy, self.LD, self.counter))
                return new_BC

        # this only triggers if condition (b) is broken.
        # once it does, it doesn't matter what order you drink it in, you just keep chugging!
        BC_copy = np.copy(self.BC)
        for _ in range(len(self.BC)):
            self.counter += 1  # add to the counter whatever is in your hands
            idx = np.random.randint(len(BC_copy))
            self.LD = BC_copy[idx]  # note whatever you drank
            BC_copy = np.delete(
                BC_copy, idx
            )  # and throw away the bottle (into recycling <3)
            self.run_data.append((BC_copy, self.H, self.LD, self.counter))

        return BC_copy

    def run(self) -> int:
        while len(self.BC) > 0:
            self.BC = self.__f()
        # ambiguity 1
        self.counter += 1

        # hold the specifics if the run was at an extreme
        if self.run_data[-1][3] > self.max_run[-1][3]:
            self.max_run = self.run_data
        elif self.run_data[-1][3] < self.min_run[-1][3]:
            self.min_run = self.run_data

        return self.counter


def bcw_scaler(N: int, K: int, seed: int, n_iter: int) -> list[int]:
    """
    I got bored of watching a singular thread burn itself to the ground so I decided to use the
    whole cluster
    """
    BCW = BeerCabinetWithdrawal(N=N, K=K, seed=seed)
    values = []
    for _ in range(n_iter):
        values.append(BCW.run())
        BCW.reset()
    return (values, BCW)


def format_line(line: list) -> str:
    __bc_l = [int(x) for x in list(line[0])]
    BC = f"BC: {__bc_l}, "
    if isinstance(line[1], str):
        return f"BC: {__bc_l}, H:, LD:, n_ops: "
    __h_l = [int(x) for x in list(line[1])]
    H = f"H: {__h_l}, "
    LD = f"LD: {int(line[2])}, "
    counter = f"n_ops: {int(line[3])}"
    return BC + H + LD + counter


def format_all_output(maxes: list, mins: list) -> None:
    if not os.path.isdir("maxes"):
        os.makedirs("maxes")
    if not os.path.isdir("mins"):
        os.makedirs("mins")
    with open(f"maxes/maxes_n_{N}_k_{K}.txt", "w+") as f:
        for idx, max_run in enumerate(maxes):
            f.write(f"Max solution {idx}\n")
            for line in max_run:
                f.write(format_line(line) + "\n")
            f.write("=========================\n")
    with open(f"mins/mins.txt_n_{N}_k_{K}", "w+") as f:
        for idx, min_run in enumerate(mins):
            f.write(f"Min solution {idx}\n")
            for line in min_run:
                f.write(format_line(line) + "\n")
            f.write("=========================\n")


if __name__ == "__main__":

    np.set_printoptions(suppress=True, formatter={"all": lambda x: str(x)})

    parser = argparse.ArgumentParser(description="Detailed beer withdrawal analysis.")
    parser.add_argument("N", type=int, help="How many beers of each class?")
    parser.add_argument("K", type=int, help="How many classes?")
    parser.add_argument("seed", type=int, help="Seed for pseudorandomness")
    parser.add_argument("n_iter", type=int, help="How many runs do you want to try?")

    args = parser.parse_args()

    N = args.N
    K = args.K
    seed = args.seed
    n_iter = args.n_iter

    if N == 2 or K == 2:
        lower_prior = 2 * (N - 1) * K + K + 1
    else:
        lower_prior = 2 * (N - 1) * (K - 1) + K + N + 1
    upper_prior = int((N - 1) * K * (K + 1) + K + (K * (N - 2) * (N - 1)) / 2 + 1)

    n_iter_per_process = int(np.rint(n_iter / cpu_count())) + 1  # seems about right
    targets = [(N, K, seed, n_iter_per_process) for _ in range(cpu_count())]
    with Pool(processes=cpu_count()) as P:
        vals_bcws = P.starmap(bcw_scaler, targets)

    # tacky but I've spent too long on this anyways
    vals = []
    bcws = []
    maxes = []
    mins = []
    for elem in vals_bcws:
        vals.append(elem[0])
        bcws.append(elem[1])
    vals = np.concatenate(vals)
    for bcw in bcws:
        maxes.append(bcw.max_run)
        mins.append(bcw.min_run)

    format_all_output(maxes=maxes, mins=mins)

    print(f"min: {min(vals)}, lower prior: {lower_prior}")
    print(f"max: {max(vals)}, upper prior: {upper_prior}")
