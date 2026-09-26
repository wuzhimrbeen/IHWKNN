"""MDGCN paper-guided runner on the frozen IHWKNN evaluation protocol.

The released MDGCN architecture is imported from the official repository.
This runner replaces its fold generation, test-epoch selection and metric code.
It therefore reports a paper-guided common-protocol baseline, not an unchanged
reproduction of the paper's own cross-validation program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
SOURCE = REPOSITORY_ROOT / "external" / "MDGCN-main"
DATA = REPOSITORY_ROOT / "data"
SUBMISSION = REPOSITORY_ROOT
DAY2 = REPOSITORY_ROOT / "experiments" / "evaluation_full_candidates"
OUT = REPOSITORY_ROOT / "experiments" / "new_baseline" / "results"


PUBLISHED = {
    "Fdataset": dict(epochs=110, hide_dim=128, decay=.99, lr=.05, layers=8, rank=4, topK=3,
                     ssl_beta=.1, ssl_reg_r=.068, ssl_reg_d=.088, wr1=.9, wr2=.1, wd1=.9, wd2=.1,
                     metareg=.19, ssl_temp=.5, new1=.9, new2=.01, eps=.3),
    "Cdataset": dict(epochs=60, hide_dim=512, decay=.99, lr=.055, layers=11, rank=6, topK=4,
                     ssl_beta=.1, ssl_reg_r=.068, ssl_reg_d=.085, wr1=.7, wr2=.3, wd1=.7, wd2=.3,
                     metareg=.15, ssl_temp=.5, new1=.9, new2=.01, eps=.6),
    "LRSSL": dict(epochs=45, hide_dim=256, decay=.99, lr=.055, layers=11, rank=6, topK=7,
                  ssl_beta=.1, ssl_reg_r=.08, ssl_reg_d=.09, wr1=.8, wr2=.2, wd1=.8, wd2=.2,
                  metareg=.15, ssl_temp=.5, new1=.75, new2=.01, eps=.2),
    "LAGCN": dict(epochs=18, hide_dim=256, decay=.99, lr=.1, layers=16, rank=4, topK=6,
                  ssl_beta=.1, ssl_reg_r=.064, ssl_reg_d=.085, wr1=.5, wr2=.5, wd1=.5, wd2=.5,
                  metareg=.16, ssl_temp=.5, new1=.25, new2=.01, eps=.3),
}
ALL_DATASETS = ["Cdataset", "Fdataset", "LRSSL", "LAGCN", "Ydataset", "SCMFDDL", "iDrug", "TLHGBI"]


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=ALL_DATASETS, default="Fdataset")
    p.add_argument("--max-folds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None,
                   help="Fixed training epochs; default uses the paper-reported dataset value")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--train-negative-ratio", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default="mdgcn_frozen_smoke")
    p.add_argument("--inner-profile-select", action="store_true",
                   help="Choose among the four published dataset profiles using only an inner training split")
    p.add_argument("--profile-probe-epochs", type=int, default=10)
    p.add_argument("--eval-repeats", type=int, default=1,
                   help="Repeated stochastic evaluation forwards for the final outer-fold model")
    p.add_argument("--reuse-profile-selection-from", type=Path, default=None,
                   help="Existing inner_profile_selection.csv used to reuse training-only profile choices")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def hetero_graph(train):
    n_drug, n_dis = train.shape
    a = sp.csr_matrix((n_drug, n_drug))
    b = sp.csr_matrix((n_dis, n_dis))
    pos = sp.csr_matrix(train != 0, dtype=np.float32)
    return sp.vstack([sp.hstack([a, pos]), sp.hstack([pos.T, b])]).tocsr()


def train_fold(dataset, train, training_zero_pairs, config, model_class, trans_data, utils,
               eval_repeats=1):
    from evaluation_protocol import fold_ranking_metrics

    seed_all(config.seed)
    rr = (utils.knn_graph(dataset.drug_similarity, config.topK) != 0) * 1.0
    dd = (utils.knn_graph(dataset.disease_similarity, config.topK) != 0) * 1.0
    rd = hetero_graph(train)
    positive = np.argwhere(train == 1).astype(np.int64)
    rows = np.concatenate([positive[:, 0], training_zero_pairs[:, 0]])
    cols = np.concatenate([positive[:, 1], training_zero_pairs[:, 1]])
    vals = np.concatenate([np.ones(len(positive)), np.zeros(len(training_zero_pairs))]).astype(np.float32)
    examples = sp.coo_matrix((vals, (rows, cols)), shape=train.shape)
    loader = DataLoader(trans_data(examples), batch_size=config.batch, shuffle=True, num_workers=0)
    model = model_class(config, *train.shape, rr, dd, rd, config.hide_dim, config.layers).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for drugs, diseases, labels in loader:
            drugs, diseases = drugs.long().cuda(), diseases.long().cuda()
            labels = labels.float().cuda()
            outputs = model(True, drugs, diseases, norm=1)
            drug_emb, dis_emb, rd_drug_all, rd_dis_all, rd_drug, rd_dis, meta_loss, all_rd = outputs
            initial, context = all_rd[0], all_rd[config.layers]
            ctx_drug, ctx_dis = torch.split(context, list(train.shape))
            ini_drug, ini_dis = torch.split(initial, list(train.shape))
            unique_drug, unique_dis = torch.unique(drugs), torch.unique(diseases)
            ssl1 = info_nce(ctx_drug[unique_drug], ini_drug[unique_drug], config.ssl_temp)
            ssl1 += info_nce(ctx_dis[unique_dis], ini_dis[unique_dis], config.ssl_temp)
            ssl_drug = utils.ssl_loss(rd_drug, drug_emb, drugs, config.ssl_temp)
            ssl_dis = utils.ssl_loss(rd_dis, dis_emb, diseases, config.ssl_temp)
            ssl_all = config.new1 * (config.ssl_reg_r * ssl_drug + config.ssl_reg_d * ssl_dis + config.new2 * ssl1)
            logits = torch.sum(rd_drug_all[drugs] * rd_dis_all[diseases], dim=1)
            loss = bce(logits, labels) + config.ssl_beta * ssl_all + config.metareg * meta_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"] * config.decay, config.min_lr)
        logging.info("%s epoch %d/%d loss=%.6f", dataset.name, epoch + 1, config.epochs, np.mean(losses))
    model.eval()
    # The released model injects random perturbations during forward even in
    # evaluation. Resetting the fold seed makes this single scoring pass
    # reproducible without changing the published architecture.
    predictions = []
    with torch.no_grad():
        ids_r = torch.arange(train.shape[0])
        ids_d = torch.arange(train.shape[1])
        for repeat in range(eval_repeats):
            seed_all(config.seed + repeat)
            outputs = model(False, ids_r, ids_d, norm=1)
            drug_final, disease_final = outputs[2], outputs[3]
            prediction = torch.sigmoid(drug_final @ disease_final.T).cpu().numpy()
            assert np.isfinite(prediction).all()
            predictions.append(prediction)
    return predictions


def info_nce(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    score = (view1 @ view2.T) / temperature
    return -torch.diag(F.log_softmax(score, dim=1)).mean()


def main():
    run = cli()
    if not torch.cuda.is_available():
        raise RuntimeError("MDGCN source requires CUDA")
    if (not 1 <= run.max_folds <= 10 or run.train_negative_ratio < 1
            or run.profile_probe_epochs < 1 or run.eval_repeats < 1):
        raise ValueError("Invalid folds or negative ratio")
    if run.dataset not in PUBLISHED and not run.inner_profile_select:
        raise ValueError("Datasets outside the official four require --inner-profile-select")
    for path in (SOURCE, SUBMISSION, DAY2 / "code"):
        sys.path.insert(0, str(path))
    from model import MODEL
    from utils import MyTransData
    import utils as mdutils
    from evaluation_protocol import fold_ranking_metrics
    from run_existing_baselines_common_protocol import stable_seed
    from src.model.ihwknn import generate_fixed_folds, load_dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = load_dataset(DATA / run.dataset)
    states = list(generate_fixed_folds(dataset.association, 10, run.seed))[:run.max_folds]
    initial_profile = run.dataset if run.dataset in PUBLISHED else "Fdataset"
    published = dict(PUBLISHED[initial_profile])
    published["epochs"] = published["epochs"] if run.epochs is None else run.epochs
    config = SimpleNamespace(dataset=run.dataset, batch=run.batch_size, seed=run.seed,
                             min_lr=.0001, **published)
    out = OUT / run.run_id / run.dataset
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "implementation_status": "MDGCN-paper-guided common-protocol port",
        "official_model_sha256": hashlib.sha256((SOURCE / "model.py").read_bytes()).hexdigest(),
        "dataset": run.dataset, "seed": run.seed, "max_folds": run.max_folds,
        "train_negative_ratio": run.train_negative_ratio, "config": vars(config),
        "selection": "fixed epochs; no test-fold epoch selection",
        "inner_profile_select": run.inner_profile_select,
        "profile_candidates": list(PUBLISHED) if run.inner_profile_select else None,
        "profile_probe_epochs": run.profile_probe_epochs if run.inner_profile_select else None,
        "reused_profile_selection": str(run.reuse_profile_selection_from) if run.reuse_profile_selection_from else None,
        "evaluation_noise": "official forward perturbation retained; deterministic repeat seeds are reset before scoring",
        "evaluation_repeats": run.eval_repeats,
    }
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Run ID already exists with another manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    partial = out / "fold_metrics.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    profile_path = out / "inner_profile_selection.csv"
    profile_rows = pd.read_csv(profile_path).to_dict("records") if profile_path.exists() else []
    sensitivity_path = out / "full_unknown_excluding_training_negatives.partial.csv"
    sensitivity_rows = pd.read_csv(sensitivity_path).to_dict("records") if sensitivity_path.exists() else []
    noise_path = out / "evaluation_noise_metrics.partial.csv"
    noise_rows = pd.read_csv(noise_path).to_dict("records") if noise_path.exists() else []
    audit_dir = out / "audit_ids"
    audit_dir.mkdir(parents=True, exist_ok=True)
    completed = {int(r["fold"]) for r in rows}
    all_zeros = np.argwhere(dataset.association == 0).astype(np.int64)
    reused_profiles = None
    if run.reuse_profile_selection_from:
        reuse_frame = pd.read_csv(run.reuse_profile_selection_from)
        reused_profiles = {
            int(row.fold): str(row.profile)
            for row in reuse_frame.itertuples(index=False)
            if str(row.selected).lower() == "true"
        }
        if set(reused_profiles) != set(range(1, run.max_folds + 1)):
            raise ValueError("Reused MDGCN profile selection does not cover every requested fold")
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
        training_zeros = all_zeros[selected]
        if run.inner_profile_select:
            if reused_profiles is not None:
                selected_profile = reused_profiles[fold]
            else:
                positive = np.argwhere(train == 1).astype(np.int64)
                split_seed = stable_seed(run.seed, dataset.name, fold, "MDGCN", "inner", "profile")
                rng = np.random.default_rng(split_seed)
                order = rng.permutation(len(positive))
                n_val = max(1, int(round(.1 * len(positive))))
                val_pos = positive[order[:n_val]]
                inner_train = train.copy()
                inner_train[val_pos[:, 0], val_pos[:, 1]] = 0
                zero_order = rng.permutation(len(all_zeros))
                n_val_zero = min(len(all_zeros), 10 * n_val)
                val_zero = all_zeros[zero_order[:n_val_zero]]
                available_zero = all_zeros[zero_order[n_val_zero:]]
                inner_zero_count = min(len(available_zero), run.train_negative_ratio * int(inner_train.sum()))
                inner_training_zeros = available_zero[:inner_zero_count]
                val_labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_zero))])
                audits = []
                for profile_name, profile_values in PUBLISHED.items():
                    probe_values = dict(profile_values)
                    probe_values["epochs"] = run.profile_probe_epochs
                    probe = SimpleNamespace(dataset=run.dataset, batch=run.batch_size, seed=run.seed,
                                            min_lr=.0001, **probe_values)
                    probe_prediction = train_fold(dataset, inner_train, inner_training_zeros,
                                                  probe, MODEL, MyTransData, mdutils)[0]
                    val_scores = np.concatenate([
                        probe_prediction[val_pos[:, 0], val_pos[:, 1]],
                        probe_prediction[val_zero[:, 0], val_zero[:, 1]],
                    ])
                    audits.append({"fold": fold, "profile": profile_name,
                                   "inner_auc": float(roc_auc_score(val_labels, val_scores)),
                                   "inner_aupr": float(average_precision_score(val_labels, val_scores)),
                                   "probe_epochs": run.profile_probe_epochs,
                                   "inner_positive_count": len(val_pos),
                                   "inner_unlabeled_count": len(val_zero)})
                selected_audit = max(audits, key=lambda x: (x["inner_auc"], x["inner_aupr"]))
                selected_profile = selected_audit["profile"]
                for audit in audits:
                    audit["selected"] = audit["profile"] == selected_profile
                profile_rows.extend(audits)
                pd.DataFrame(profile_rows).to_csv(profile_path, index=False)
            final_values = dict(PUBLISHED[selected_profile])
            if run.epochs is not None:
                final_values["epochs"] = run.epochs
            fold_config = SimpleNamespace(dataset=run.dataset, batch=run.batch_size, seed=run.seed,
                                          min_lr=.0001, **final_values)
            logging.info("%s fold %d selected MDGCN profile %s", dataset.name, fold, selected_profile)
        else:
            selected_profile = run.dataset
            fold_config = config
        predictions = train_fold(dataset, train, training_zeros, fold_config, MODEL,
                                 MyTransData, mdutils, eval_repeats=run.eval_repeats)
        prediction = predictions[0]
        training_zero_ids = training_zeros[:, 0] * train.shape[1] + training_zeros[:, 1]
        np.save(audit_dir / f"fold_{fold:02d}_training_zero_ids.npy", training_zero_ids)
        for protocol, candidate in fold_candidates.items():
            rows.append({"dataset": dataset.name,
                         "method": "MDGCN-adapted" if run.inner_profile_select else "MDGCN-paper-guided",
                         "fold": fold, "selected_profile": selected_profile,
                         "candidate_protocol": protocol, "runtime_seconds": time.time() - started,
                         "training_zero_count": count, "train_negative_ratio": run.train_negative_ratio,
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
            "method": "MDGCN-adapted" if run.inner_profile_select else "MDGCN-paper-guided",
            "fold": fold,
            "selected_profile": selected_profile,
            "excluded_training_negative_count": int(len(full_candidate.unlabeled_ids) - len(filtered_ids)),
            **fold_ranking_metrics(prediction, filtered_candidate),
        })
        pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
        for repeat, repeat_prediction in enumerate(predictions, 1):
            noise_rows.append({
                "dataset": dataset.name,
                "method": "MDGCN-adapted" if run.inner_profile_select else "MDGCN-paper-guided",
                "fold": fold,
                "evaluation_repeat": repeat,
                "evaluation_seed": fold_config.seed + repeat - 1,
                **fold_ranking_metrics(repeat_prediction, full_candidate),
            })
        pd.DataFrame(noise_rows).to_csv(noise_path, index=False)
        pd.DataFrame(rows).to_csv(partial, index=False)
        logging.info("Completed %s fold %d/%d in %.1fs", dataset.name, fold, run.max_folds, time.time() - started)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(out / "full_unknown_excluding_training_negatives.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(out / "evaluation_noise_metrics.csv", index=False)


if __name__ == "__main__":
    main()
