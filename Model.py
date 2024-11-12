import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import copy

# Process the data and divide the dataset into training set, validation set and testing set
def create_data(data_source, modulation, target, M=False, percentage_data=1):
    global df1
    df = pd.read_csv(data_source, index_col=0)
    df = df[df['Validity'] != 0]
    scaler = MinMaxScaler()

    # # 对 P 和 Vref 列进行归一化
    df[['P', 'Vref']] = scaler.fit_transform(df[['P', 'Vref']])

    if modulation == 'EPS1' and target == 'Ptotal':
        df1 = df.drop(['D2', 'D0', 'Validity', 'ipk', 'irms', 'Vo', 'nZVS'], axis=1)
    elif modulation == 'EPS1' and target == 'nZVS':
        df1 = df.drop(['D2', 'D0', 'Validity', 'ipk', 'irms', 'Vo', 'Ptotal'], axis=1)
    elif modulation == 'EPS2' and target == 'Ptotal':
        df1 = df.drop(['D1', 'D0', 'Validity', 'ipk', 'irms', 'Vo', 'nZVS'], axis=1)
    elif modulation == 'EPS2' and target == 'nZVS':
        df1 = df.drop(['D1', 'D0', 'Validity', 'ipk', 'irms', 'Vo', 'Ptotal'], axis=1)
    df1 = df1.reset_index(drop=True)

    if "Set" not in df1.columns:
        if target=='Ptotal':
            np.random.seed(2025)
            df1["Set"] = np.random.choice(["train", "valid", "test"], p=[.6, .2, .2], size=(df1.shape[0],))
        if target=='nZVS':
            np.random.seed(6020)
            df1["Set"] = np.random.choice(["train", "valid", "test"], p=[.6, .2, .2], size=(df1.shape[0],))
    print(df1)
    unused_feat = ['Set']
    features = [col for col in df1.columns if col not in unused_feat + [target]]

    train_indices = df1[df1.Set == "train"].index
    valid_indices = df1[df1.Set == "valid"].index
    test_indices = df1[df1.Set == "test"].index
    new_train_indices = None
    X_train = df1[features].values[train_indices]
    X_valid = df1[features].values[valid_indices]
    X_test = df1[features].values[test_indices]
    if target == 'Ptotal':
        y_train = df1[target].values[train_indices].reshape(-1, 1)
        y_valid = df1[target].values[valid_indices].reshape(-1, 1)
        y_test = df1[target].values[test_indices].reshape(-1, 1)
    else:
        y_train = (df1[target].values[train_indices] - 4) / 2
        y_valid = (df1[target].values[valid_indices] - 4) / 2
        y_test = (df1[target].values[test_indices] - 4) / 2
    if M:
        train_indices_df = pd.DataFrame(train_indices, columns=['index'])

        # 20% random selection from train_indices
        new_train_indices_df = train_indices_df.sample(
            frac=percentage_data)  # Select 20 per cent of the data, you can adjust the value of frac

        # Extract index and convert back to Int64Index
        new_train_indices = pd.Index(new_train_indices_df['index']).astype('int64')

        X_train = df1[features].values[new_train_indices]
        y_train = df1[target].values[new_train_indices].reshape(-1, 1)

    return X_train, y_train, X_valid, y_valid, X_test, y_test



#Create a TabNet model, where the tunable parameters of the model can be found in pytorch-tabnet - https://dreamquark-ai.github.io/tabnet/
def Model_TabNet(X_train, y_train, X_valid, y_valid, target, n_d=16, n_a=16,epochs=300, patience=100, batch_size=512,
                 virtual_batch_size=128):
    if target == 'Ptotal':
        clf = TabNetRegressor(n_d=n_d, n_a=n_a)
        metric = ['mse']
    elif target == 'nZVS':
        clf = TabNetClassifier(n_d=n_d, n_a=n_a)
        metric = ['accuracy']
    max_epochs = epochs if not os.getenv("CI", False) else 2


    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=['train', 'valid'],
        eval_metric=metric,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size, virtual_batch_size=virtual_batch_size,
        num_workers=0,
    )

    return clf

#Build a residual neural network
class Single_Residual(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, final=False):
        super(Single_Residual, self).__init__()
        self.linear_layer = nn.Linear(input_size, hidden_size)
        self.final = final
        if not self.final:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            self.dropout = nn.Dropout(dropout)
        self.residual_maker = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.residual_maker.requires_grad_(False)
        if hidden_size <= input_size:
            self.residual_maker[torch.randperm(input_size)[:hidden_size], torch.arange(0, hidden_size)] = 1
        else:
            self.residual_maker[torch.arange(0, input_size), torch.randperm(hidden_size)[:input_size]] = 1

    def forward(self, x):
        if self.final:
            h1 = self.linear_layer(x)
        else:
            h1 = F.relu(self.linear_layer(x))
        return h1

class Residual_DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, num_classes):
        super(Residual_DNN, self).__init__()
        assert isinstance(hidden_sizes, list), 'the "hidden_sizes" should be list '
        hidden_num = len(hidden_sizes)
        self.residuals = [f"self.residual{i}" for i in range(hidden_num)]
        for i in range(hidden_num):
            if i == hidden_num - 1:
                exec(self.residuals[i] + "=Single_Residual(input_size,hidden_sizes[i],dropout,True)")
            else:
                exec(self.residuals[i] + "=Single_Residual(input_size,hidden_sizes[i],dropout)")
            input_size = hidden_sizes[i]
        self.output_layer = nn.Linear(input_size, num_classes)

    def forward(self, x):
        for residual in self.residuals:
            x = eval(residual)(x)
        x = self.output_layer(x)
        return x


def Model_NN(X_train, y_train, X_valid, y_valid, target):


    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_valid = torch.tensor(X_valid).float()
    y_valid = torch.tensor(y_valid).float()

    if target=='nZVS':
        y_train=y_train.long()
        y_valid=y_valid.long()
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    if target == 'Ptotal':
        num_classes = 1
        criterion = nn.MSELoss()
        mode = 'min'
        best_evaluation = float('inf')
    else:
        num_classes = 3
        criterion = nn.CrossEntropyLoss()
        mode = 'max'
        best_evaluation = 0

    input_size = X_train.shape[1]
    hidden_sizes = [128, 64, 32]
    dropout = 0.1
    model = Residual_DNN(input_size, hidden_sizes, dropout, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=10, factor=0.5, verbose=True)

    epochs = 300
    early_stopping_patience = 100
    best_model_state = None
    patience = 0

    for epoch in range(epochs):
        all_predictions_train = []
        all_targets_train = []
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if target == 'Ptotal':
                all_predictions_train.extend(outputs.detach().numpy())
                all_targets_train.extend(targets.detach().numpy())
                evaluation_train = mean_squared_error(all_targets_train, all_predictions_train)
            else:
                _, predicted = torch.max(outputs, 1)
                all_predictions_train.extend(predicted.detach().numpy())
                all_targets_train.extend(targets.detach().numpy())
                evaluation_train = accuracy_score(all_targets_train, all_predictions_train)
                # Evaluate model performance
        model.eval()
        all_predictions_valid = []
        all_targets_valid = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                if target == 'Ptotal':
                    all_predictions_valid.extend(outputs.numpy())
                    all_targets_valid.extend(targets.numpy())
                else:
                    _, predicted = torch.max(outputs, 1)
                    all_predictions_valid.extend(predicted.numpy())
                    all_targets_valid.extend(targets.numpy())

        if target == 'Ptotal':
            evaluation_valid = mean_squared_error(all_targets_valid, all_predictions_valid)
            print(
                f"Epoch [{epoch + 1}/{epochs}]  - Train MSE: {evaluation_train:.4f} - Validation MSE: {evaluation_valid:.4f}")
            scheduler.step(evaluation_valid)

            if evaluation_valid < best_evaluation:
                best_evaluation = evaluation_valid
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

        if target == 'nZVS':
            evaluation_valid = accuracy_score(all_targets_valid, all_predictions_valid)
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Train ACC: {evaluation_train:.4f} - Validation ACC: {evaluation_valid:.4f}")
            scheduler.step(evaluation_valid)

            if evaluation_valid > best_evaluation:
                best_evaluation = evaluation_valid
                best_model_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

        if patience > early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs,with best Validation MSE:{best_evaluation:.4f}.")
            break

    model.load_state_dict(best_model_state)

    return model


def Model_prevailing(Model,X_train,y_train,target):
    """This function is used to construct the model by a variety of algorithms, supported algorithms include RandomForest, XGBoost and SVM. Hyperparameter tuning methods for each algorithm can be found on the following website
    RandomForest:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    XGBoost:https://xgboost.readthedocs.io/en/latest/index.html
    SVM:https://scikit-learn.org/stable/modules/svm.html
    """

    if Model=='RandomForest' :
        param_grid = {
            'n_estimators': [50, 100,200],
            'max_depth': [5,8,10],
            'min_samples_split': [2,3,4],
                    }
        if target=='Ptotal':
            model=RandomForestRegressor()
            scoring='neg_mean_squared_error'
        else:
            model=RandomForestClassifier()
            scoring='accuracy'

    if Model=='XGBoost' :
        param_grid = {
            'n_estimators': [50,100,120,150,200],
            'subsample': [0.5,0.8,1],
            'max_depth': [3,5,8,10],
    #        'eta':[0.2,0.3,0.4]
        }
        if target=='Ptotal':
            model=xgb.XGBRegressor()
            scoring='neg_mean_squared_error'
        else:
            model=xgb.XGBClassifier()
            scoring='accuracy'

    if Model=='SVM':
        param_grid = {
            'C': [30,50,100],
            'kernel': ['linear',  'rbf'],
            'degree':[3,4,5]
        }
        if target=='Ptotal':
            model=SVR()
            scoring='neg_mean_squared_error'
        else:
            model=SVC()
            scoring='accuracy'

    # Hyperparameter tuning with grid searc
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model,best_params

def score(Model,target,X_test,y_test):
    if type(Model) == Residual_DNN:
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()
        if target=='nZVS':
            y_test=y_test.long()
        test_dataset = TensorDataset(X_test,y_test)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
        Model.eval()
        #Calculate the MSE and MAE on the test set
        all_predictions_test = []
        all_targets_test = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = Model(inputs)
                if target=='Ptotal':
                    all_predictions_test.extend(outputs.numpy())
                    all_targets_test.extend(targets.numpy())
                    test_score = mean_squared_error(all_predictions_test,all_targets_test)
                else:
                    _, predicted = torch.max(outputs, 1)
                    all_predictions_test.extend(predicted.numpy())
                    all_targets_test.extend(targets.numpy())
                    test_score = accuracy_score(all_predictions_test,all_targets_test)
    else:

        preds=Model.predict(X_test)
        if target=='Ptotal':
            test_score = mean_squared_error(y_pred=preds, y_true=y_test)
        else:
            test_score=accuracy_score(y_pred=preds, y_true=y_test)
    return test_score