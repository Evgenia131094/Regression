from sklearn.metrics import mean_squared_error, \
    r2_score, \
    explained_variance_score, \
    max_error, \
    mean_absolute_error, \
    median_absolute_error, \
    mean_absolute_percentage_error
import torch


def test(model, y_test, test_loader, device):

    mse = 0
    evs = 0
    me = 0
    mae = 0
    medianae = 0
    r2 = 0
    mape = 0
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)

            y_batch = y_batch.cpu().numpy()
            y_test_pred = model(x_batch)

            mse += mean_squared_error(y_test_pred.cpu().numpy(), y_batch)
            evs += explained_variance_score(y_test_pred.cpu().numpy(), y_batch)
            me += max_error(y_test_pred.cpu().numpy(), y_batch)
            mae += mean_absolute_error(y_test_pred.cpu().numpy(), y_batch)
            medianae += median_absolute_error(y_test_pred.cpu().numpy(), y_batch)
            r2 += r2_score(y_test_pred.cpu().numpy(), y_batch)
            mape += mean_absolute_percentage_error(y_test_pred.cpu().numpy(), y_batch)

    print(" Mean Squared Error :", mse / len(y_test))
    print(" Explained Variance Score :", evs / len(y_test))
    print(" Max error :", me / len(y_test))
    print(" Mean absolute error :", mae / len(y_test))
    print(" Median absolute error :", medianae / len(y_test))
    print(" R2 score :", r2 / len(y_test))
    print(" Mean absolute percentage error :", mape / len(y_test))
