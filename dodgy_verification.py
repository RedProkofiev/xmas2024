from multiprocessing import Pool, cpu_count

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class BeerCabinetWithdrawal:
    """
    Highly sophisticated class that spews out N * f(B | BC) given a case N and K.
    Run it a couple hundred thousand/million times and believe in the central limit theorem - 
    For any given N and K above 1, it should converge at a figure stated in the proof.
    """
    def __init__(self, N: int, K: int, seed: int = 42, debug: bool = False):
        self.rng = np.random.default_rng(seed=seed)
        self.N = N
        self.K = K
        self.debug = debug

        self.reset()

        self.max_run = [(self.BC, "", "", -np.inf)]
        self.min_run = [(self.BC, "", "", np.inf)]

        self.H = []

    def __pre_c(self, H_seq) -> int:
        for idx in H_seq:
            if (self.BC[idx] in self.H) and (self.BC[idx] != self.LD):
                # self.LD = self.BC[idx]
                self.H.append(self.BC[idx])
                return idx
            self.H.append(self.BC[idx])
        return -1
    
    def __post_c(self, H_seq) -> int:
        return H_seq[0]
    
    # ((my logic is missing individual operations because once a duplicate is found it doesn't add them to hand))

    # treat f as a recursive function that creates its own BC
    def __f(self) -> list[int]:
        # given a random retrieval order H...
        self.H.clear() # return beers to BC to begin a new round...
        H_seq = np.random.permutation(np.arange(len(self.BC)))
        if not self.post_phase:
            # H will be full
            idx = self.__pre_c(H_seq)
            if idx == -1:
                self.post_phase = True
                # extract one from current state of H
                # note that this does not increment, as the withdrawal
                # has already taken place.
                if self.BC[H_seq[0]] == self.LD:
                    idx = H_seq[1] # can be out of bounds?
                    # how?
                else:
                    idx = H_seq[0]
        else:
            idx = self.__post_c(H_seq)
            self.H.append(self.BC[idx])

        # you have to actually add all the things to your hands though stupid!!!
        self.counter += len(self.H) # add to the counter whatever is in your hands
        # if len(self.H) > 1 and self.post_phase:  # edge case
        #     for i 
        self.LD = self.BC[idx] # note whatever you drank
        new_BC = np.delete(self.BC, idx) # and throw away the bottle (into recycling <3)

        # reference errors?
        h_copy = np.copy(self.H)
        self.run_data.append((new_BC, h_copy, self.LD, self.counter))
        return new_BC
    
    def run(self) -> int:
        # individual run
        while len(self.BC) > 0:
            if self.debug:
                print_h = [int(x) for x in self.H]
                print(f"self.BC: {self.BC}")
                print(f"self.H: {print_h}")
                print(f"self.counter: {self.counter}")
                if self.LD:
                    print(f"self.LD: {self.LD}")         
            self.BC = self.__f()
        self.counter += 1 # gotta actually find you have the empty set

        # analytica
        if self.run_data[-1][3] > self.max_run[-1][3]:
            self.max_run = self.run_data
        elif self.run_data[-1][3] < self.min_run[-1][3]:
            self.min_run = self.run_data

        return self.counter
    
    def reset(self) -> None:
        self.counter = 0
        self.LD = -1
        self.post_phase = False
        self.BC = np.repeat(np.arange(1, self.N + 1), self.K)
        # syntax: BC, H, removed beer, counter
        self.run_data = [(self.BC, "", "", self.counter)]


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


if __name__ == '__main__':

    np.set_printoptions(suppress=True, formatter={'all':lambda x: str(x)})

    N = int(sys.argv[1])
    K = int(sys.argv[2])
    seed = int(sys.argv[3])
    n_iter = int(sys.argv[4])

    if seed == 0:
        debug = True
        seed = np.random.randint(100000)
    else:
        debug = False

    """
        This best-case prior is built of four components:
        1. (N - 1) * K * 2, pre-rule (C) being activated
            In this best of all worlds, you only have to pull two drinks to drink one,
            rather than More
        2. K operations to go through the post-rule (C) section
        3. In this best of all worlds, you never pull a drink you can't drink.  Lovely!
        4. + 1 to verify the set is empty (you must stick your hands in BC)
    """
    lower_prior = (N - 1) * K * 2 + K + 1
    
    """
        This worst-case prior is built of four components:
        1. (N - 1)K(K + 1), pre-rule (C) being activated
        2. K operations to go through the post-rule (C) section
        3. (K * (N - 2) * (N - 1))/2 for (B, 2.) or "no-two-same"
        4. + 1 to verify the set is empty (you must stick your hands in BC)
    """
    upper_prior = int((N - 1) * K * (K + 1) + K + (K * (N - 2) * (N - 1))/2 + 1)

    if not debug:
        n_iter_per_process = int(np.rint(n_iter/cpu_count())) + 1 # seems about right
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

        # nevermind it gets worse
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
    else:
        BCW = BeerCabinetWithdrawal(N=N, K=K, seed=seed, debug=debug)
        vals = []
        for _ in range(n_iter):
            vals.append(BCW.run())
            BCW.reset()

    print(f"min: {min(vals)}, lower prior: {lower_prior}")
    print(f"max: {max(vals)}, upper prior: {upper_prior}")
