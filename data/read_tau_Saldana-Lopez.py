import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ebl_data_import():
    redshifts = 0  # [DL], redshifts
    energies = 0  # [TeV], background photon energies
    tau = 0  # [DL], optical density

    with open("Saldana-Lopez/tau_saldana-lopez21.out", "r") as file_name:
        data = file_name.readlines()
        m = len(data) - 4
        energies = np.zeros(m)
        for i, line in enumerate(data):
            if i < 2 or i == 3:
                continue

            elif i == 2:
                line_i = line.strip().split(", ")
                n = len(line_i)
                redshifts = np.zeros(n)
                for j in range(n):
                    if j == 0:
                        redshifts[j] = float(line_i[j][-5:])
                    else:
                        redshifts[j] = float(line_i[j])
                    tau = np.zeros([n, m])

            else:
                line_i = line.strip().split(" ")
                counter = 0
                for j in range(len(line_i)):
                    if line_i[j] == '':
                        continue
                    if j == 0:
                        energies[i-4] = float(line_i[j])
                        counter += 1
                    else:
                        tau[counter-1, i-4] = float(line_i[j])
                        counter += 1

    data = pd.DataFrame(tau, columns=energies, index=redshifts)
    print(data)

    plt.pcolormesh(data.columns, data.index, data)
    plt.colorbar()
    plt.show()

    return


if __name__ == '__main__':
    ebl_data_import()
