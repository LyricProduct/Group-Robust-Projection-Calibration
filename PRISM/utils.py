import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip


def encode_text(model, prompts, device):
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


def classify_images(
    args,
    model,
    text_embeddings,
    test_loader,
    P=None,
    description="Classifying",
):
    correct = 0
    total = 0
    all_y = None
    all_preds = None
    all_metadata = None
    if P is not None:
        P = P.to(text_embeddings.dtype)
        text_embeddings = text_embeddings @ P

    with torch.no_grad():
        for images, labels, metadata in tqdm(test_loader, desc=description):
            images = images.to(args.device)
            labels = labels.to(args.device)

            image_embeddings = model.encode_image(images)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            image_embeddings = image_embeddings.to(text_embeddings.dtype)

            if P is not None:
                P = P.to(image_embeddings.dtype)
                image_embeddings = image_embeddings @ P

            similarity = image_embeddings @ text_embeddings.T

            preds = similarity.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if all_y is None:
                all_y = labels
                all_preds = preds
                all_metadata = metadata
            else:
                all_y = torch.cat((all_y, labels))
                all_preds = torch.cat((all_preds, preds))
                all_metadata = torch.cat((all_metadata, metadata))

    print(f"Total samples processed: {total}")
    print(f"Correct predictions: {correct}")
    accuracy = correct / total * 100
    return accuracy, [all_y, all_preds, all_metadata]


def evaluate_nontarget(args, encoding_coll, label_coll, text_embeddings, P=None):
    correct = 0
    total = 0
    if P is not None:
        P = P.to(text_embeddings.dtype)
        text_embeddings = text_embeddings @ P

    with torch.no_grad():
        for image_encodings, labels in tqdm(
            zip(encoding_coll, label_coll),
            total=len(encoding_coll),
            desc="Evaluating non-target classes",
        ):
            image_encodings = image_encodings.to(args.device).to(text_embeddings.dtype)
            labels = labels.to(args.device)

            if P is not None:
                image_encodings = image_encodings @ P
                preds = (image_encodings @ text_embeddings.T).argmax(dim=1)
            else:
                preds = get_zeroshot_predictions(
                    image_encodings, text_embeddings, args, temperature=100.0
                )
                preds = torch.from_numpy(preds).to(args.device)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


def accuracy_by_subgroup(pred, y, spurious):
    assert len(pred) == len(y) == len(spurious), "All inputs must have the same length."

    counts_correct = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    counts_total = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    for p, true_label, sp in zip(pred, y, spurious):
        key = (true_label, sp)
        counts_total[key] += 1
        if p == true_label:
            counts_correct[key] += 1

    for subgroup in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        corr = counts_correct[subgroup]
        tot = counts_total[subgroup]
        if tot > 0:
            acc = corr / tot
            print(
                f"Accuracy for y={subgroup[0]}, spurious={subgroup[1]}: {acc:.3f} "
                f"({corr}/{tot} correct)"
            )
        else:
            print(
                f"Accuracy for y={subgroup[0]}, spurious={subgroup[1]}: N/A (no samples)"
            )


def print_per_class(args, eval_results):
    if args.dataset == "celeba":
        print("Accuracy for subgroup")
        print(eval_results[0]["acc_y:notblond_male:0"])
        print(eval_results[0]["acc_y:notblond_male:1"])
        print(eval_results[0]["acc_y:blond_male:0"])
        print(eval_results[0]["acc_y:blond_male:1"])
    elif args.dataset == "waterbirds":
        print("Accuracy for subgroup")
        print(eval_results[0]["acc_y:landbird_background:water"])
        print(eval_results[0]["acc_y:landbird_background:land"])
        print(eval_results[0]["acc_y:waterbird_background:water"])
        print(eval_results[0]["acc_y:waterbird_background:land"])


def embedding_regularization(
    embeddings,
    transformed_embeddings,
    reg_type="l2",
    reg_lambda=1e-3,
    transformer=None,
    gram_lambda=1e-3,
    ortho_lambda=1e-3,
):
    embeddings_neg1 = embeddings[0]
    embeddings_neg2 = embeddings[1]
    emb_neg1_transf = transformed_embeddings[0]
    emb_neg2_transf = transformed_embeddings[1]

    reg_loss = torch.zeros(1, device=embeddings_neg1.device)[0]

    if reg_type in ("l2", "l1", "cos"):
        if reg_type == "l2":
            reg_loss += reg_lambda * (embeddings_neg1 - emb_neg1_transf).pow(2).mean()
            reg_loss += reg_lambda * (embeddings_neg2 - emb_neg2_transf).pow(2).mean()

        elif reg_type == "l1":
            reg_loss += reg_lambda * (embeddings_neg1 - emb_neg1_transf).abs().mean()
            reg_loss += reg_lambda * (embeddings_neg2 - emb_neg2_transf).abs().mean()

        elif reg_type == "cos":
            reg_loss += (
                reg_lambda
                * (
                    1 - F.cosine_similarity(embeddings_neg1, emb_neg1_transf, dim=-1)
                ).mean()
            )
            reg_loss += (
                reg_lambda
                * (
                    1 - F.cosine_similarity(embeddings_neg2, emb_neg2_transf, dim=-1)
                ).mean()
            )

        return reg_loss

    if reg_type in ("gram", "gramorth"):
        W = torch.cat([embeddings_neg1.squeeze(1), embeddings_neg2.squeeze(1)], dim=0)
        Wt = torch.cat([emb_neg1_transf.squeeze(1), emb_neg2_transf.squeeze(1)], dim=0)

        G_orig = W @ W.T
        G_trans = Wt @ Wt.T

        gram_loss = (G_trans - G_orig).pow(2).mean()
        reg_loss += gram_lambda * gram_loss

    if reg_type in ("orth", "gramorth"):
        if transformer is None:
            raise ValueError(
                "transformer must be provided if reg_type requires orthonormal regularization."
            )

        A = transformer.transformer.weight

        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        orth_loss = (A.T @ A - I).pow(2).mean()

        reg_loss += ortho_lambda * orth_loss

    return reg_loss


def contrastive_loss(embeddings, margin=0.5):
    embeddings_neg1 = F.normalize(embeddings[0], dim=-1)
    embeddings_neg2 = F.normalize(embeddings[1], dim=-1)
    embeddings_pos1 = F.normalize(embeddings[2], dim=-1)
    embeddings_pos2 = F.normalize(embeddings[3], dim=-1)

    sim_neg = F.cosine_similarity(embeddings_neg1, embeddings_neg2, dim=-1)
    sim_pos = F.cosine_similarity(embeddings_pos1, embeddings_pos2, dim=-1)

    positive_loss = 1 - sim_pos
    negative_loss = torch.relu(sim_neg - margin)

    loss = positive_loss.mean() + negative_loss.mean()

    return loss


def sentence_list_to_embedding(args, model, sentences):
    st = [clip.tokenize(sentence).to(args.device) for sentence in sentences]
    se = [model.encode_text(token) for token in st]
    se = [embedding / embedding.norm(dim=-1, keepdim=True) for embedding in se]
    return torch.stack(se)


def train_transformation(args, model, textloader, transformer):
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=args.lr, weight_decay=args.wd
    )
    for epoch in range(args.epochs):
        total_loss = 0
        iter = 0

        for senetnces_list in tqdm(textloader):
            optimizer.zero_grad()

            embeddigs_list = [
                sentence_list_to_embedding(args, model, sentences)
                for sentences in senetnces_list
            ]

            transformed_embeddings = [
                transformer(embeddings.float()) for embeddings in embeddigs_list
            ]

            loss = contrastive_loss(transformed_embeddings)
            if args.reg_lambda > 0 and args.reg_type is not None:
                reg_loss = embedding_regularization(
                    embeddigs_list,
                    transformed_embeddings,
                    args.reg_type,
                    args.reg_lambda,
                    transformer,
                    args.reg_lambda,
                    args.reg_lambda,
                )
                loss += reg_loss

            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            iter += 1

        print(f"Epoch: {epoch}, Loss: {total_loss/iter}")


def classify_with_embeddings(
    image_embeddings, text_embeddings, args, temperature=100.0
):
    with torch.no_grad():
        _image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )

        _text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        cross = _image_embeddings @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)

    return predicted.cpu().numpy()


def get_zeroshot_predictions(key_embeddings, text_embeddings, args, temperature=100.0):
    predictions = classify_with_embeddings(
        key_embeddings, text_embeddings, args, temperature=100.0
    )

    return predictions
