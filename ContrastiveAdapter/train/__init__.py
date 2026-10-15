import numpy as np
import torch


def prep_metrics(dataloader, args):
    running_loss = 0.0
    correct = 0
    total = 0

    targets_s = dataloader.dataset.targets_all["spurious"]
    targets_t = dataloader.dataset.targets_all["target"]
    correct_by_groups = np.zeros([args.num_classes, args.num_classes])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)
    return (
        running_loss,
        correct,
        total,
        targets_s,
        targets_t,
        correct_by_groups,
        total_by_groups,
        losses_by_groups,
    )


def train_epoch(model, dataloader, optimizer, criterion, args, evaluate=True):
    (
        running_loss,
        correct,
        total,
        targets_s,
        targets_t,
        correct_by_groups,
        total_by_groups,
        losses_by_groups,
    ) = prep_metrics(dataloader, args)

    try:
        model.to_device(args.device)
    except:
        model.to(args.device)

    model.train()
    model.zero_grad()

    for batch_ix, data in enumerate(dataloader):
        inputs, labels, data_ix = data

        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        labels_target = [targets_t[ix] for ix in data_ix]
        labels_spurious = [targets_s[ix] for ix in data_ix]

        y_outputs = model(inputs)
        loss = criterion(y_outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, predicted = torch.max(y_outputs.data, 1)
        total += labels.size(0)
        all_correct = predicted == labels
        correct += all_correct.sum().item()
        running_loss += loss.item()

        for ix, s in enumerate(labels_spurious):
            y = labels_target[ix]
            correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
            total_by_groups[int(y)][int(s)] += 1

        inputs = inputs.cpu()
        loss = loss.cpu()

    try:
        model.to_device(torch.device("cpu"))
    except:
        model.cpu()

    return running_loss, correct, total, correct_by_groups, total_by_groups


def evaluate(model, dataloader, criterion, args, save_embeddings=False):
    (
        running_loss,
        correct,
        total,
        targets_s,
        targets_t,
        correct_by_groups,
        total_by_groups,
        losses_by_groups,
    ) = prep_metrics(dataloader, args)

    try:
        model.to_device(args.device)
    except:
        model.to(args.device)
    model.eval()

    data_embeddings = []

    with torch.no_grad():
        for batch_ix, data in enumerate(dataloader):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            labels_target = [targets_t[dix] for dix in data_ix]
            labels_spurious = [targets_s[ix] for ix in data_ix]

            if save_embeddings and "adapter" in args.train_method:
                outputs, embeddings = model(inputs, return_hidden=True)
                embeddings = embeddings.cpu()
                data_embeddings.append(embeddings)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_correct = (predicted == labels).detach().cpu()
            correct += all_correct.sum().item()
            running_loss += loss.item()

            for ix, s in enumerate(labels_spurious):
                y = labels_target[ix]
                correct_by_groups[int(y)][int(s)] += all_correct[ix].item()
                total_by_groups[int(y)][int(s)] += 1

            inputs = inputs.to(torch.device("cpu"))
            outputs = outputs.to(torch.device("cpu"))
            loss = loss.to(torch.device("cpu"))
            del outputs
            del inputs
            del labels
            del loss

    try:
        model.to_device(torch.device("cpu"))
    except:
        model.cpu()

    if save_embeddings:
        data_embeddings = torch.vstack(data_embeddings)
        return (
            running_loss,
            correct,
            total,
            correct_by_groups,
            total_by_groups,
            data_embeddings,
        )

    return running_loss, correct, total, correct_by_groups, total_by_groups
