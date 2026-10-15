import numpy as np


def summarize_acc(correct_by_groups, total_by_groups, stdout=True, return_groups=False):
    all_correct = 0
    all_total = 0
    min_acc = 101.0
    min_correct_total = [None, None]
    groups_accs = np.zeros([len(correct_by_groups), len(correct_by_groups[-1])])
    if stdout:
        print("Accuracies by groups:")
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            acc = a_group / total_by_groups[yix][aix] * 100
            groups_accs[yix][aix] = acc

            if acc < min_acc and total_by_groups[yix][aix] > 0:
                min_acc = acc
                min_correct_total[0] = a_group
                min_correct_total[1] = total_by_groups[yix][aix]
            if stdout:
                print(
                    f"{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}"
                )
            all_correct += a_group
            all_total += total_by_groups[yix][aix]
    if stdout:
        average_str = f"Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}"
        robust_str = f"Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}"
        print("-" * len(average_str))
        print(average_str)
        print(robust_str)
        print("-" * len(average_str))

    avg_acc = all_correct / all_total * 100

    if return_groups:
        return avg_acc, min_acc, groups_accs
    return avg_acc, min_acc
