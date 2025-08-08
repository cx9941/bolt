from configs import args
import os
import copy
if os.path.exists(args.metric_file):
    exit()
if os.path.exists(args.case_path + f'/OpenMax_preds.npy'):
    exit()

import os
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve
from src.pytorch_ood.detector import (
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
    OpenMax,
    ASH,
    SHE
)
from src.pytorch_ood.utils import OODMetrics, fix_random_seed, custom_metrics

import numpy as np
import logging

from model import Model
from load_dataset import train_dataset, dataset_in_test, dataset_out_test, collate_batch, loader_in_train, loader_in_test, test_loader, eval_loader, loader_in_eval


if __name__ == '__main__':
    fix_random_seed(args.seed)
    model = Model(args).to(args.device)
    with torch.no_grad():
        model.eval()
        preds = []
        golds = []
        logits = []
        features = []
        for batch in tqdm(test_loader):
            y = batch['labels'].to(args.device)
            batch = {i:v.to(args.device) for i,v in batch.items() if i!= 'labels'}
            logit = model(batch)
            feature = model.features(batch)
            pred = logit.max(dim=1).indices
            preds.append(pred)
            logits.append(logit)
            golds.append(y)
            features.append(feature)
        logits = torch.concat(logits).detach().cpu().numpy()
        preds = torch.concat(preds).detach().cpu().numpy()
        golds = torch.concat(golds).detach().cpu().numpy()
        features = torch.concat(features).detach().cpu().numpy()
        ID_metrics = custom_metrics(preds, golds)
        
        logging.info(f"Test Accuracy: {ID_metrics['macro avg']}")

    logging.info("STAGE 2: Creating OOD Detectors")

    detectors = {}
    detectors["SHE"] = SHE(model=model.features, head=model.fc)
    detectors["OpenMax"] = OpenMax(model)
    detectors["ViM"] = ViM(model.features, d=64, w=model.fc.weight, b=model.fc.bias)
    detectors["Entropy"] = Entropy(model)
    detectors["Mahalanobis"] = Mahalanobis(model.features, eps=0.0)
    detectors["KLMatching"] = KLMatching(model)
    detectors["MaxSoftmax"] = MaxSoftmax(model)
    detectors["EnergyBased"] = EnergyBased(model)
    detectors["MaxLogit"] = MaxLogit(model)

    logging.info(f"> Fitting {len(detectors)} detectors")

    for name, detector in detectors.items():
        logging.info(f"--> Fitting {name}")
        detector.fit(loader_in_train, device=args.device)

    logging.info(f"STAGE 3: Evaluating {len(detectors)} detectors.")
    results = []

    with torch.no_grad():
        for detector_name, detector in detectors.items():
            logging.info(f"> Evaluating {detector_name}")
            metrics = OODMetrics()
            scores = []
            for batch in tqdm(test_loader):
                y = batch['labels'].to(args.device)
                batch = {i:v.to(args.device) for i,v in batch.items() if i!= 'labels'}
                score = detector(batch)
                metrics.update(score, y)
                scores.append(score)

            r = {"Detector": detector_name}
            r.update(metrics.compute())
            scores = torch.concat(scores).detach().cpu().numpy()
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

            if detector_name == 'Vim':
                new_logits = np.concatenate([logits, scores.reshape(scores.shape[0], 1)], axis=1)
                new_preds = new_logits.argmax(axis=1)
                new_preds[new_preds==logits.shape[-1]] = -1
                preds = new_preds
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
            else:
                r.update(custom_metrics(preds, golds, norm_scores))
                final_preds = copy.deepcopy(preds)
                final_preds[norm_scores > 0.5] = -1

            results.append(r)
            np.save(args.case_path + f'/{detector_name}_preds.npy', final_preds)
            np.save(args.case_path + f'/{detector_name}_golds.npy', golds)
            np.save(args.case_path + f'/{detector_name}_features.npy', features)

    df = pd.DataFrame(results)
    mean_scores = df.groupby(["Detector"]).mean() * 100
    logging.info(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))
    mean_scores.to_csv(args.metric_file,  sep='\t')