import numpy as np
import torch

from os.path import join


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


def summarize_acc_from_predictions(predictions, dataloader, args, stdout=True):
    targets_t = dataloader.dataset.targets_all["target"]
    targets_s = dataloader.dataset.targets_all["spurious"]

    correct_by_groups = np.zeros([args.num_classes, args.num_classes])
    total_by_groups = np.zeros(correct_by_groups.shape)

    all_correct = predictions == targets_t
    for ix, s in enumerate(targets_s):
        y = targets_t[ix]
        correct_by_groups[int(y)][int(s)] += all_correct[ix]
        total_by_groups[int(y)][int(s)] += 1
    return summarize_acc(correct_by_groups, total_by_groups, stdout=stdout)


def log_metrics(
    train_metrics, val_metrics, test_metrics, epoch, dataset_ix=0, args=None
):
    assert args is not None
    train_loss, correct, total, correct_by_groups, total_by_groups = train_metrics
    train_avg_acc, train_min_acc = summarize_acc(
        correct_by_groups, total_by_groups, stdout=args.verbose
    )
    val_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
    val_avg_acc, val_min_acc = summarize_acc(
        correct_by_groups, total_by_groups, stdout=args.verbose
    )
    test_loss, correct, total, correct_by_groups, total_by_groups = test_metrics
    test_avg_acc, test_min_acc = summarize_acc(
        correct_by_groups, total_by_groups, stdout=args.verbose
    )

    args.results_dict["epoch"].append(epoch)
    args.results_dict["dataset_ix"].append(dataset_ix)
    args.results_dict["train_loss"].append(train_loss)
    args.results_dict["train_avg_acc"].append(train_avg_acc)
    args.results_dict["train_robust_acc"].append(train_min_acc)

    args.results_dict["val_loss"].append(val_loss)
    args.results_dict["val_avg_acc"].append(val_avg_acc)
    args.results_dict["val_robust_acc"].append(val_min_acc)

    args.results_dict["test_loss"].append(test_loss)
    args.results_dict["test_avg_acc"].append(test_avg_acc)
    args.results_dict["test_robust_acc"].append(test_min_acc)

    train_metrics = (train_loss, train_avg_acc, train_min_acc)
    val_metrics = (val_loss, val_avg_acc, val_min_acc)
    return train_metrics, val_metrics


def get_model_path(args, prefix, train_method):
    return join(
        args.model_dir,
        f"{prefix}-{train_method}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}.pt",
    )


def process_validation_metrics(
    model,
    val_metrics,
    epoch,
    train_method,
    args,
    best_loss=None,
    best_avg_acc=None,
    best_robust_acc=None,
    best_loss_epoch=None,
    best_avg_acc_epoch=None,
    best_robust_acc_epoch=None,
):
    val_loss, correct, total, correct_by_groups, total_by_groups = val_metrics
    avg_acc, min_acc = summarize_acc(correct_by_groups, total_by_groups, stdout=False)

    if val_loss < best_loss or epoch == 0:
        best_loss_epoch = epoch
        best_loss = val_loss
        args.best_loss_model_path = get_model_path(args, "m-best_loss", train_method)
        torch.save(model.state_dict(), args.best_loss_model_path)
    args.results_dict["best_loss_epoch"].append(best_loss_epoch)

    if avg_acc > best_avg_acc or epoch == 0:
        best_avg_acc_epoch = epoch
        best_avg_acc = avg_acc
        args.best_avg_acc_model_path = get_model_path(args, "m-best_acc", train_method)
        torch.save(model.state_dict(), args.best_avg_acc_model_path)
    args.results_dict["best_acc_epoch"].append(best_avg_acc_epoch)

    if min_acc > best_robust_acc or epoch == 0:
        best_robust_acc_epoch = epoch
        best_robust_acc = min_acc
        args.best_robust_acc_model_path = get_model_path(
            args, "m-best_acc_r", train_method
        )
        torch.save(model.state_dict(), args.best_robust_acc_model_path)
    args.results_dict["best_robust_acc_epoch"].append(best_robust_acc_epoch)

    return (
        (best_loss, best_loss_epoch),
        (best_avg_acc, best_avg_acc_epoch),
        (best_robust_acc, best_robust_acc_epoch),
    )
