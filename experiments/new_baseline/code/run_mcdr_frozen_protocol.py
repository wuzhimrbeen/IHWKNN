"""MCDR paper-guided runner on frozen IHWKNN folds and candidates.

Feature modes:
  original   Use the released precomputed DDI prediction matrix (available for
             Fdataset, Cdataset and LRSSL only).
  similarity Replace that drug-side DDI feature branch with the supplied drug
             similarity graph/features. This is explicitly an adapted no-DDI
             model and can be evaluated on all eight IHWKNN datasets.

The official architecture is retained in original mode. Similarity mode uses a
one-line isolated port; the official source tree remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
SOURCE = REPOSITORY_ROOT / "external" / "MCDR-main"
PORT = REPOSITORY_ROOT / "experiments" / "new_baseline" / "ports" / "MCDR_no_DDI" / "model.py"
DATA = REPOSITORY_ROOT / "data"
SUBMISSION = REPOSITORY_ROOT
DAY2 = REPOSITORY_ROOT / "experiments" / "evaluation_full_candidates"
OUT = REPOSITORY_ROOT / "experiments" / "new_baseline" / "results"
DATASETS = ["Cdataset", "Fdataset", "LRSSL", "LAGCN", "Ydataset", "SCMFDDL", "iDrug", "TLHGBI"]
DDI_FILES = {
    "Fdataset": SOURCE / "G_ddi_prediction_matrix.csv",
    "Cdataset": SOURCE / "C_ddi_prediction_matrix.csv",
    "LRSSL": SOURCE / "L_ddi_prediction_matrix.csv",
}


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default="Fdataset")
    p.add_argument("--feature-mode", choices=["original", "similarity"], default="similarity")
    p.add_argument("--max-folds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--inner-select", action="store_true",
                   help="Select the epoch using only an inner split of the outer training fold")
    p.add_argument("--epoch-candidates", default="10,25,50,100,200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-negative-ratio", type=int, default=1)
    p.add_argument("--score-batch-size", type=int, default=65536)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--run-id", default="mcdr_frozen_smoke")
    p.add_argument("--reuse-fold-metrics", type=Path, default=None,
                   help="Existing fold_metrics.csv from which selected epochs are reused")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates(dataset, fold):
    from evaluation_protocol import CandidateSet
    checkpoint = DAY2 / "checkpoints" / "day2_full_candidates_seed42" / f"{dataset.name}_candidate_protocol.npz"
    with np.load(checkpoint) as payload:
        assert tuple(payload["association_shape"].tolist()) == dataset.association.shape
        pos = payload[f"fold_{fold:02d}_test_positive_ids"].astype(np.int64)
        zero = np.flatnonzero(dataset.association.ravel() == 0).astype(np.int64)
        result = {"full_unknown": CandidateSet("full_unknown", pos, zero)}
        for ratio in (1, 5, 10, 50):
            result[f"1:{ratio}"] = CandidateSet(
                f"1:{ratio}", pos,
                payload[f"fold_{fold:02d}_ratio_{ratio}_unlabeled_ids"].astype(np.int64),
            )
    return result


def model_args(dataset, epochs, seed, runtime_device):
    n_drug, n_disease = dataset.association.shape
    return SimpleNamespace(
        # The released DGL wheel in the current environment is CPU-only. Keep
        # the GCMC layer's internal device unset on CPU; model.to() below moves
        # ordinary tensors and parameters to the selected runtime device.
        device=None if runtime_device == "cpu" else torch.device("cuda:0"),
        runtime_device=runtime_device, seed=seed, model_activation="tanh", dropout=.3,
        gcn_agg_units=840, gcn_agg_accum="sum", gcn_out_units=75,
        train_max_iter=epochs, train_grad_clip=1., gcn_agg_norm_symm=True,
        num_neighbor=12, nhid1=500, nhid2=75, train_lr=.01, layers=2,
        intra=.2, inter=.2, num_hidden=75, num_proj_hidden1=100,
        num_proj_hidden2=150, share_param=True, rating_vals=np.array([0., 1.]),
        src_in_units=n_drug + n_disease + 3,
        dst_in_units=n_drug + n_disease + 3,
        fdim_drug=n_drug, fdim_disease=n_disease,
    )


def make_builder(dataset, feature_mode, data_module, runtime_device):
    builder = data_module.DrugDataLoader.__new__(data_module.DrugDataLoader)
    builder._name = dataset.name
    builder._device = torch.device(runtime_device)
    builder._symm = True
    builder.num_neighbor = 12
    builder._num_drug, builder._num_disease = dataset.association.shape
    builder.drug_sim_features = np.asarray(dataset.drug_similarity, dtype=np.float32)
    builder.disease_sim_features = np.asarray(dataset.disease_similarity, dtype=np.float32)
    if feature_mode == "original":
        if dataset.name not in DDI_FILES:
            raise ValueError(f"No released MCDR DDI matrix for {dataset.name}")
        builder.ddi = pd.read_csv(DDI_FILES[dataset.name], index_col=0).values.astype(np.float32)
        if builder.ddi.shape != (builder._num_drug, builder._num_drug):
            raise ValueError("DDI matrix shape mismatch")
    else:
        # Only needed so the unchanged graph-builder can initialize. The
        # isolated no-DDI model port consumes drug_graph/drug_sim_features.
        builder.ddi = builder.drug_sim_features
    builder.possible_rel_values = np.array([0., 1.], dtype=np.float32)
    builder._generate_feat()
    builder.drug_graph, builder.disease_graph, builder.ddi_graph = builder._generate_feat_graph()
    return builder


def relation_graphs(builder, train, zeros):
    positive = np.argwhere(train == 1).astype(np.int64)
    pairs = np.concatenate([positive, zeros], axis=0)
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(zeros))]).astype(np.float32)
    frame = pd.DataFrame({"disease_id": pairs[:, 0], "drug_id": pairs[:, 1], "values": labels})
    rating_pairs, values = builder._generate_pair_value(frame)
    enc = builder._generate_enc_graph(rating_pairs, values, add_support=True)
    dec = builder._generate_dec_graph(rating_pairs)
    return enc, dec, torch.from_numpy(values), positive


def train_and_score(dataset, train, zeros, args, builder, model_module, score_batch_size):
    seed_all(args.seed)
    enc, dec, truth, positive = relation_graphs(builder, train, zeros)
    device = torch.device(args.runtime_device)
    enc, dec, truth = enc.int().to(device), dec.int().to(device), truth.to(device)
    drug_graph = builder.drug_graph.to(device)
    disease_graph = builder.disease_graph.to(device)
    ddi_graph = builder.ddi_graph.to(device)
    drug_sim = torch.from_numpy(builder.drug_sim_features).float().to(device)
    disease_sim = torch.from_numpy(builder.disease_sim_features).float().to(device)
    ddi_feat = torch.from_numpy(builder.ddi).float().to(device)
    drug_feat, disease_feat = builder.drug_feature, builder.disease_feature
    model = model_module.Net(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(args.train_max_iter):
        model.train()
        pred, *_ = model(enc, dec, drug_graph, drug_sim, drug_feat, ddi_graph, ddi_feat,
                         disease_graph, disease_sim, disease_feat, False)
        loss = loss_fn(pred.squeeze(-1), truth)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()
        logging.info("%s %s epoch %d/%d loss=%.6f", dataset.name,
                     "no-DDI" if model_module.__name__.endswith("no_ddi") else "original-DDI",
                     epoch + 1, args.train_max_iter, float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        # One dummy edge obtains the trained node representations once.
        dummy = builder._generate_dec_graph((np.array([0]), np.array([0]))).int().to(device)
        _, drug_out, _, disease_out, _ = model(
            enc, dummy, drug_graph, drug_sim, drug_feat, ddi_graph, ddi_feat,
            disease_graph, disease_sim, disease_feat, False,
        )
        n_drug, n_disease = train.shape
        flat = np.empty(n_drug * n_disease, dtype=np.float32)
        for start in range(0, len(flat), score_batch_size):
            stop = min(start + score_batch_size, len(flat))
            ids = np.arange(start, stop, dtype=np.int64)
            pairs = (ids // n_disease, ids % n_disease)
            graph = builder._generate_dec_graph(pairs).int().to(device)
            flat[start:stop] = torch.sigmoid(model.decoder(graph, drug_out, disease_out)).reshape(-1).cpu().numpy()
            del graph
    prediction = flat.reshape(train.shape)
    assert np.isfinite(prediction).all()
    return prediction


def score_pairs(model, builder, enc, graph_inputs, pairs, device):
    drug_graph, drug_sim, drug_feat, ddi_graph, ddi_feat, disease_graph, disease_sim, disease_feat = graph_inputs
    with torch.no_grad():
        dummy = builder._generate_dec_graph((np.array([0]), np.array([0]))).int().to(device)
        _, drug_out, _, disease_out, _ = model(
            enc, dummy, drug_graph, drug_sim, drug_feat, ddi_graph, ddi_feat,
            disease_graph, disease_sim, disease_feat, False,
        )
        graph = builder._generate_dec_graph((pairs[:, 0], pairs[:, 1])).int().to(device)
        return torch.sigmoid(model.decoder(graph, drug_out, disease_out)).reshape(-1).cpu().numpy()


def select_epoch_inner(dataset, outer_train, args, builder, model_module, epoch_candidates,
                       train_negative_ratio, fold):
    """Select training duration without reading any outer-test label or score."""
    from run_existing_baselines_common_protocol import stable_seed

    device = torch.device(args.runtime_device)
    positives = np.argwhere(outer_train == 1).astype(np.int64)
    split_seed = stable_seed(args.seed, dataset.name, fold, "MCDR", "inner", "positive_split")
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(positives))
    n_val = max(1, int(round(.1 * len(positives))))
    val_pos = positives[order[:n_val]]
    inner_train = outer_train.copy()
    inner_train[val_pos[:, 0], val_pos[:, 1]] = 0
    original_zeros = np.argwhere(dataset.association == 0).astype(np.int64)
    neg_order = rng.permutation(len(original_zeros))
    n_val_zero = min(len(original_zeros), 10 * n_val)
    val_zero = original_zeros[neg_order[:n_val_zero]]
    available = original_zeros[neg_order[n_val_zero:]]
    n_train_zero = min(len(available), train_negative_ratio * int(inner_train.sum()))
    train_zero = available[:n_train_zero]
    enc, dec, truth, _ = relation_graphs(builder, inner_train, train_zero)
    enc, dec, truth = enc.int().to(device), dec.int().to(device), truth.to(device)
    graph_inputs = (
        builder.drug_graph.to(device),
        torch.from_numpy(builder.drug_sim_features).float().to(device), builder.drug_feature,
        builder.ddi_graph.to(device), torch.from_numpy(builder.ddi).float().to(device),
        builder.disease_graph.to(device),
        torch.from_numpy(builder.disease_sim_features).float().to(device), builder.disease_feature,
    )
    seed_all(args.seed)
    model = model_module.Net(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoint_set = set(epoch_candidates)
    validation_pairs = np.concatenate([val_pos, val_zero], axis=0)
    validation_labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_zero))])
    audit = []
    for epoch in range(1, max(epoch_candidates) + 1):
        model.train()
        pred, *_ = model(enc, dec, *graph_inputs, False)
        loss = loss_fn(pred.squeeze(-1), truth)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()
        if epoch in checkpoint_set:
            model.eval()
            scores = score_pairs(model, builder, enc, graph_inputs, validation_pairs, device)
            auc = float(roc_auc_score(validation_labels, scores))
            aupr = float(average_precision_score(validation_labels, scores))
            audit.append({"fold": fold, "epoch": epoch, "inner_auc": auc,
                          "inner_aupr": aupr, "inner_positive_count": len(val_pos),
                          "inner_unlabeled_count": len(val_zero)})
            logging.info("%s inner fold %d epoch %d AUC=%.4f AUPR=%.4f",
                         dataset.name, fold, epoch, auc, aupr)
    selected = max(audit, key=lambda x: (x["inner_auc"], x["inner_aupr"], -x["epoch"]))["epoch"]
    return int(selected), audit


def main():
    run = cli()
    if run.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 1 <= run.max_folds <= 10 or run.epochs < 1 or run.train_negative_ratio < 1:
        raise ValueError("Invalid run controls")
    epoch_candidates = sorted({int(x) for x in run.epoch_candidates.split(",") if x.strip()})
    if not epoch_candidates or epoch_candidates[0] < 1:
        raise ValueError("Invalid epoch candidates")
    for path in (SOURCE, SUBMISSION, DAY2 / "code"):
        sys.path.insert(0, str(path))
    import dgl.function as dgl_fn
    if not hasattr(dgl_fn, "copy_src"):
        # DGL >=1 renamed the 0.6 API used by the released source.
        dgl_fn.copy_src = lambda src, out: dgl_fn.copy_u(src, out)
    import data as data_module
    from evaluation_protocol import fold_ranking_metrics
    from run_existing_baselines_common_protocol import stable_seed
    from src.model.ihwknn import generate_fixed_folds, load_dataset

    model_path = SOURCE / "model.py" if run.feature_mode == "original" else PORT
    module_name = "mcdr_original" if run.feature_mode == "original" else "mcdr_no_ddi"
    model_module = load_module(model_path, module_name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = load_dataset(DATA / run.dataset)
    states = list(generate_fixed_folds(dataset.association, 10, run.seed))[:run.max_folds]
    config = model_args(dataset, run.epochs, run.seed, run.device)
    builder = make_builder(dataset, run.feature_mode, data_module, run.device)
    method = "MCDR-paper-guided" if run.feature_mode == "original" else "MCDR-adapted-no-DDI"
    out = OUT / run.run_id / run.dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "implementation_status": method + " common-protocol port",
        "official_model_sha256": hashlib.sha256((SOURCE / "model.py").read_bytes()).hexdigest(),
        "executed_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "dataset": run.dataset, "feature_mode": run.feature_mode, "seed": run.seed,
        "max_folds": run.max_folds, "epochs": run.epochs,
        "device": run.device,
        "inner_select": run.inner_select,
        "epoch_candidates": epoch_candidates if run.inner_select else None,
        "reused_fold_metrics": str(run.reuse_fold_metrics) if run.reuse_fold_metrics else None,
        "train_negative_ratio": run.train_negative_ratio,
        "score_batch_size": run.score_batch_size,
        "selection": "inner 90/10 training-fold split; AUC then AUPR" if run.inner_select else "fixed experimental budget; test fold never selects epochs",
        "no_ddi_adaptation": "drug similarity graph/features replace released DDI-prediction branch" if run.feature_mode == "similarity" else None,
    }
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Run ID already exists with another manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    partial = out / "fold_metrics.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    inner_path = out / "inner_epoch_selection.csv"
    inner_rows = pd.read_csv(inner_path).to_dict("records") if inner_path.exists() else []
    sensitivity_path = out / "full_unknown_excluding_training_negatives.partial.csv"
    sensitivity_rows = pd.read_csv(sensitivity_path).to_dict("records") if sensitivity_path.exists() else []
    audit_dir = out / "audit_ids"
    audit_dir.mkdir(parents=True, exist_ok=True)
    completed = {int(r["fold"]) for r in rows}
    all_zeros = np.argwhere(dataset.association == 0).astype(np.int64)
    reused_epochs = None
    if run.reuse_fold_metrics:
        reuse_frame = pd.read_csv(run.reuse_fold_metrics)
        reused_epochs = {
            int(fold): int(group["selected_epoch"].iloc[0])
            for fold, group in reuse_frame.groupby("fold")
        }
        if set(reused_epochs) != set(range(1, run.max_folds + 1)):
            raise ValueError("Reused MCDR epoch selection does not cover every requested fold")
    for fold, (train, test_pairs) in enumerate(states, 1):
        if fold in completed:
            continue
        started = time.time()
        fold_candidates = candidates(dataset, fold)
        expected = test_pairs[:, 0] * train.shape[1] + test_pairs[:, 1]
        assert np.array_equal(np.sort(expected), np.sort(fold_candidates["full_unknown"].positive_ids))
        count = min(len(all_zeros), run.train_negative_ratio * int(train.sum()))
        neg_seed = stable_seed(run.seed, dataset.name, fold, "COMMON", "COMMON", "outer_negative")
        selected = np.random.default_rng(neg_seed).choice(len(all_zeros), size=count, replace=False)
        zeros = all_zeros[selected]
        fold_config = SimpleNamespace(**vars(config))
        if reused_epochs is not None:
            selected_epoch = reused_epochs[fold]
            fold_config.train_max_iter = selected_epoch
        elif run.inner_select:
            selected_epoch, audit = select_epoch_inner(
                dataset, train, fold_config, builder, model_module,
                epoch_candidates, run.train_negative_ratio, fold,
            )
            inner_rows.extend(audit)
            pd.DataFrame(inner_rows).to_csv(inner_path, index=False)
            fold_config.train_max_iter = selected_epoch
        else:
            selected_epoch = run.epochs
        prediction = train_and_score(dataset, train, zeros, fold_config, builder, model_module, run.score_batch_size)
        training_zero_ids = zeros[:, 0] * train.shape[1] + zeros[:, 1]
        np.save(audit_dir / f"fold_{fold:02d}_training_zero_ids.npy", training_zero_ids)
        for protocol, candidate in fold_candidates.items():
            rows.append({"dataset": dataset.name, "method": method, "fold": fold,
                         "candidate_protocol": protocol, "runtime_seconds": time.time() - started,
                         "training_zero_count": count, "train_negative_ratio": run.train_negative_ratio,
                         "selected_epoch": selected_epoch,
                         **fold_ranking_metrics(prediction, candidate)})
        full_candidate = fold_candidates["full_unknown"]
        filtered_ids = np.setdiff1d(full_candidate.unlabeled_ids, training_zero_ids,
                                    assume_unique=False)
        from evaluation_protocol import CandidateSet
        filtered_candidate = CandidateSet(
            "full_unknown_excluding_training_negatives",
            full_candidate.positive_ids,
            filtered_ids,
        )
        sensitivity_rows.append({
            "dataset": dataset.name,
            "method": method,
            "fold": fold,
            "selected_epoch": selected_epoch,
            "excluded_training_negative_count": int(len(full_candidate.unlabeled_ids) - len(filtered_ids)),
            **fold_ranking_metrics(prediction, filtered_candidate),
        })
        pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
        pd.DataFrame(rows).to_csv(partial, index=False)
        logging.info("Completed %s %s fold %d/%d in %.1fs", dataset.name, method,
                     fold, run.max_folds, time.time() - started)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "full_unknown_excluding_training_negatives.csv", index=False)


if __name__ == "__main__":
    main()
