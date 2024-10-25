#
# This code is created by Hsin-Yuan Huang (https://momohuang.github.io/).
# For more details, see the accompany paper:
# "Predicting Many Properties of a Quantum System from Very Few Measurements".
# This Python version is slower than the C++ version. (there are less code optimization)
# But it should be easier to understand and build upon.
#
# Yong Wang (2023-10-12): I have reduced the duplication of computations for some functions, speed up computation
import sys
import random
import math


def randomized_classical_shadow(num_total_measurements, system_size):
    #
    # Implementation of the randomized classical shadow
    #
    #    num_total_measurements: int for the total number of measurement rounds
    #    system_size: int for how many qubits in the quantum system
    #
    measurement_procedure = []
    for t in range(num_total_measurements):
        single_round_measurement = [random.choice(
            [1, 2, 0]) for i in range(system_size)]
        measurement_procedure.append(single_round_measurement)
    return measurement_procedure


def derandomized_classical_shadow(target_obs, target_locs, num_of_measurements_per_observable, system_size, weight=None):
    #
    # Implementation of the derandomized classical shadow
    #
    #     num_of_measurements_per_observable: int for the number of measurement for each observable
    #     system_size: int for how many qubits in the quantum system
    #     weight: None or a list of coefficients for each observable
    #             None -- neglect this parameter
    #             a list -- modify the number of measurements for each observable by the corresponding weight
    #
    n_observable = len(target_obs)
    if weight is None:
        weight = [1.0] * n_observable
    assert (len(weight) == n_observable)

    sum_log_value = 0
    sum_cnt = 0

    def cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift=0):
        eta = 0.9  # a hyperparameter subject to change
        nu = 1 - math.exp(-eta / 2)

        nonlocal sum_log_value
        nonlocal sum_cnt

        cost = 0
        for i, zipitem in enumerate(zip(num_of_measurements_so_far, num_of_matches_needed_in_this_round)):
            measurement_so_far, matches_needed = zipitem
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                continue

            if system_size < matches_needed:
                V = eta / 2 * measurement_so_far
            else:
                V = eta / 2 * measurement_so_far - \
                    math.log(1 - nu / (3 ** matches_needed))
            cost += math.exp(-V / weight[i] - shift)

            sum_log_value += V / weight[i]
            sum_cnt += 1

        return cost

    def match_up(qubit_i, dice_roll_pauli, single_obs, single_locs):
        if qubit_i in single_locs:
            idx = single_locs.index(qubit_i)
            if single_obs[idx] != dice_roll_pauli:
                return -1
            else:
                return 1
        else:
            return 0

    num_of_measurements_so_far = [0] * n_observable
    measurement_procedure = []

    for repetition in range(num_of_measurements_per_observable * n_observable):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_needed_in_this_round = [len(P) for P in target_obs]
        single_round_measurement = []

        shift = sum_log_value / sum_cnt if sum_cnt > 0 else 0
        sum_log_value = 0.0
        sum_cnt = 0

        for qubit_i in range(system_size):
            cost_of_outcomes = dict([(1, 0), (2, 0), (0, 0)])

            results_all = []
            for dice_roll_pauli in [1, 2, 0]:
                # Assume the dice rollout to be "dice_roll_pauli"
                results = []
                for i, single_obs in enumerate(target_obs):
                    result = match_up(
                        qubit_i, dice_roll_pauli, single_obs, target_locs[i])
                    results.append(result)
                    if result == -1:
                        # impossible to measure
                        num_of_matches_needed_in_this_round[i] += 100 * (
                            system_size + 10)
                    if result == 1:
                        # match up one Pauli X/Y/Z
                        num_of_matches_needed_in_this_round[i] -= 1

                cost_of_outcomes[dice_roll_pauli] = cost_function(
                    num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift=shift)

                # Revert the dice roll
                for i in range(n_observable):
                    result = results[i]
                    if result == -1:
                        # impossible to measure
                        num_of_matches_needed_in_this_round[i] -= 100 * (
                            system_size + 10)
                    if result == 1:
                        # match up one Pauli X/Y/Z
                        num_of_matches_needed_in_this_round[i] += 1

                results_all.append(results)

            for idx, dice_roll_pauli in enumerate([1, 2, 0]):
                if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                    continue
                # The best dice roll outcome will come to this line
                single_round_measurement.append(dice_roll_pauli)

                for i in range(n_observable):
                    result = results_all[idx][i]
                    if result == -1:
                        # impossible to measure
                        num_of_matches_needed_in_this_round[i] += 100 * (
                            system_size+10)
                    if result == 1:
                        # match up one Pauli X/Y/Z
                        num_of_matches_needed_in_this_round[i] -= 1
                break

        measurement_procedure.append(single_round_measurement)

        for i in range(n_observable):
            # finished measuring all qubits
            if num_of_matches_needed_in_this_round[i] == 0:
                num_of_measurements_so_far[i] += 1

        success = 0
        for i in range(n_observable):
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                success += 1

        if success == n_observable:
            break

    return measurement_procedure
