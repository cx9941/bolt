from configs import args
import os
import pandas as pd
import torch
from tqdm import tqdm
import logging
from model import Model
from load_dataset import test_loader
from sklearn.cluster import KMeans
from utils import clustering_score
import numpy as np

if __name__ == '__main__':
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


    kmeans = KMeans(n_clusters=golds.max() + 1)
    kmeans.fit(features)
    y_kmeans = kmeans.predict(features)
    metrics = clustering_score(golds, y_kmeans, np.arange(args.num_labels))

    mean_scores = pd.DataFrame([metrics])
    mean_scores.to_csv(args.metric_file,  sep='\t', index=None)