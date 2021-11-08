import os
import argparse

import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from data_preparation import clean_table
from dataset import RegressionDataset
from model import RegModel
from train import train_one_epoch, val_one_epoch
from test import test
from loss_classes import ContinuousLoss_SL1, ContinuousLoss_L2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='mode')

    parser.add_argument('--method', type=str, default='DNN', choices=['DNN', 'XGB'],
                        help='computing method')

    parser.add_argument('--data_path', type=str, default="data", required=False,
                        help='Path to csv files')
    parser.add_argument('--log_path', type=str, default="log", required=False,
                        help='Path to log directory')
    parser.add_argument('--model_path', type=str, default='models', help='Name of the directory to save models')

    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--continuous_loss_type', type=str, default='MSE', choices=['L2', 'Smooth L1', 'MSE'],
                        help='type of continuous loss')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--batch_size', type=int, default=2000)  # default=1000 use batch size = double(categorical emotion classes)
    parser.add_argument('--pretrained_models', type=bool, default=False)
    # Generate args
    args = parser.parse_args()
    return args


def check_folders(args):
    folders = [args.model_path, args.log_path]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def run():
    # configuration
    args = parse_args()

    DEVICE = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    check_folders(args)

    # prepare data for regression
    path = os.path.join(os.getcwd(), "data")

    try:
        missing_values = ["n/a", "na", "-", "н/п", "#VALUE!", "нет пробы", "", " "]
        df = pd.read_csv(os.path.join(path, "all_data_2.csv"), na_values=missing_values, sep=",")
        # df_train = pd.read_csv(os.path.join(path, "data70_1.csv"), na_values=missing_values, sep=",")
        # df_val = pd.read_csv(os.path.join(path, "data-10.csv"), na_values=missing_values, sep=",")
        # df_test = pd.read_csv(os.path.join(path, "data-20.csv"), na_values=missing_values, sep=",")
        # df_train.fillna(-1, inplace=True)
        # df_val.fillna(-1, inplace=True)
        # df_test.fillna(-1, inplace=True)
        df.fillna(-1, inplace=True)

    except FileNotFoundError:
        print("Preparing data")
        print("Data are ready! Please, restart this app")
        return 0

    print("Data are ready")
    print(df.head())

    Y = df["Суммарный расход ФК"]
    X = df.drop(['Суммарный расход ФК'], axis=1)

    # y_train = df_train["FKSum"]
    # X_train = df_train.drop(['FKSum'], axis=1)
    #
    # y_val = df_val["FKSum"]
    # X_val = df_val.drop(['FKSum'], axis=1)
    # in_features = len(X_train.columns)
    #
    # y_test = df_test["FKSum"]
    # X_test = df_test.drop(['FKSum'], axis=1)

    # X = clean_table(X)
    #
    # Splitting data on train, val, test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, Y, test_size=0.3, random_state=69)  # Split train into train-val
    X_test, y_test = np.array(X_test.astype(float)), np.array(y_test.astype(float))

    if args.mode == "train":
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.34, random_state=21)

        # preparing model
        model = RegModel(in_features=len(X_train.columns))
        model.to(DEVICE)

        X_train, y_train = np.array(X_train.astype(float)), np.array(y_train.astype(float))
        X_val, y_val = np.array(X_val.astype(float)), np.array(y_val.astype(float))

        if args.method == "XGB":
            print("XGB ....")

            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import cross_val_score, KFold
            from sklearn.metrics import mean_squared_error, \
                r2_score, \
                explained_variance_score, \
                max_error, \
                mean_absolute_error, \
                median_absolute_error, \
                mean_absolute_percentage_error
            import matplotlib.pyplot as plt
            import xgboost as xgb

            xgbr = xgb.XGBRegressor(verbosity=0)
            print(xgbr)

            xgbr.fit(X_train, y_train)

            score = xgbr.score(X_train, y_train)

            print("Training score: ", score)

            # - cross validataion
            scores = cross_val_score(xgbr, X_train, y_train, cv=5)
            print("Mean cross-validation score: %.2f" % scores.mean())

            kfold = KFold(n_splits=10, shuffle=True)
            kf_cv_scores = cross_val_score(xgbr, X_train, y_train, cv=kfold)
            print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

            ypred = xgbr.predict(X_test)
            mse = mean_squared_error(y_test, ypred)
            evs = explained_variance_score(y_test, ypred)
            mae = mean_absolute_error(y_test, ypred)
            medianae = median_absolute_error(y_test, ypred)
            r2 = r2_score(y_test, ypred)
            mape = mean_absolute_percentage_error(y_test, ypred)

            print(" Mean Squared Error :", mse)
            print(" Explained Variance Score :", evs)
            print(" Max error :", mae)
            print(" Mean absolute error :", mae)
            print(" Median absolute error :", medianae)
            print(" R2 score :", r2)
            print(" Mean absolute percentage error :", mape)
            print("RMSE: %.2f" % (mse ** (1 / 2.0)))

            x_ax = range(len(y_test))
            plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
            plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
            plt.legend()
            plt.savefig("grapf.png") #znach loss
            #plt.show()

        elif args.method == "DNN":
            print("DNN ....")
            print("Dataset and Dataloader preparation...")

            train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
            test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

            if args.continuous_loss_type == 'Smooth L1':
                criterion = ContinuousLoss_SL1()
            elif args.continuous_loss_type == 'L2':
                criterion = ContinuousLoss_L2()
            else:
                criterion = torch.nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0, last_epoch=-1)

            print("Start training...")

            loss_stats = {
                'train': [],
                "val": []
            }

            train_writer = SummaryWriter(os.path.join(args.log_path, "train"))
            val_writer = SummaryWriter(os.path.join(args.log_path, "val"))

            for e in range(1, EPOCHS + 1):

                train_loss, model, optimizer, scheduler = train_one_epoch(model, optimizer, scheduler, criterion, train_loader, DEVICE)
                val_loss = val_one_epoch(model, criterion, val_loader, DEVICE)

                loss_stats['train'].append(train_loss / len(train_loader))
                loss_stats['val'].append(val_loss / len(val_loader))

                print(f'Epoch {e + 0:03}: | Train Loss: {loss_stats["train"][-1]:.5f} | Val Loss: {loss_stats["val"][-1]:.5f}')

                train_writer.add_scalar('losses/loss', loss_stats["train"][-1], e)
                val_writer.add_scalar('losses/loss', loss_stats["val"][-1], e)

                if e % 5 == 0:
                    print("Check Point")
                    torch.save(model, os.path.join(args.model_path, 'model_{0}.pth'.format(e)))

            print("Start Testing...")
            test(model, y_test, test_loader, DEVICE)
        else:
            print("Unknown method")
    else:
        model = torch.load(os.path.join(args.model_path, 'model_3000.pth'))
        test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
        print("Start Testing...")
        test(model, y_test, test_loader, DEVICE)



if __name__ == '__main__':
    run()

