import numpy as np
from dataloader import Artist20Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

def train(root):
    
    train_dataset = Artist20Dataset(root, mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    X_list, y_list = [], []

    for wavs, melspecs, mfccs, labels in tqdm(train_dataloader):
        mfccs = mfccs.squeeze().numpy()
        labels = labels.squeeze().numpy()
        # print(mfccs.shape)

        mfccs_mean = mfccs.mean(axis=1)   # (batch_size, n_mfcc)
        mfccs_std  = mfccs.std(axis=1)    # (batch_size, n_mfcc)
        mfccs_vec = np.concatenate([mfccs_mean, mfccs_std])  # (batch_size, n_mfcc*2)

        X_list.append(mfccs_vec)
        y_list.append(labels)

    X = np.stack((X_list))
    y = np.stack((y_list))
    # print(f"X_list shape: {X.shape}")
    # print(f"y_list shape: {y.shape}")

    # clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    # clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,    
        max_depth=6,           
        subsample=0.8,          
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(set(y)),
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)

    return clf, X, y 

if __name__ == "__main__":
    # for test
    root = "dataset"
    clf, X, y = train(root)
    y_pred = clf.predict(X)
    print("Train Acc:", accuracy_score(y, y_pred))
