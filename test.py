from sklearn.metrics import mean_squared_error, r2_score
import torch


def test(model, y_test, test_loader, device):

    mse = 0

    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.cpu().numpy()
            y_test_pred = model(x_batch)
            mse += mean_squared_error(y_test_pred.cpu().numpy(), y_batch)

    print("Mean Squared Error :", mse / len(y_test))
