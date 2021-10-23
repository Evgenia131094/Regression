import torch


def train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device):
    # TRAINING
    train_epoch_loss = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:

        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        # train_loss = criterion(y_train_pred, y_train_batch)
        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        train_loss.backward()
        train_epoch_loss += train_loss.item()
        optimizer.step()
    scheduler.step()
    return train_epoch_loss / len(train_loader), model, optimizer, scheduler


def val_one_epoch(model, criterion, val_loader, device):
    # VALIDATION
    with torch.no_grad():
        val_epoch_loss = 0
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()
    return val_epoch_loss / len(val_loader)