import torch
import math
import sys
import time
import datetime
from .tools import AverageMeter, accuracy
from thop import profile
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score


def train_one_epoch(epoch, iterator, data, model, device, optimizer, criterion, tensorboard, start_time, args, r,
                    adj_matrix=None, adj_perturbation=None, record=False):
    print('--------------------------Start training at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)

    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    criterion = criterion.to(device)

    model.train()
    data, data_labels = data
    steps = data.shape[0] // args.batch_size + 1 if data.shape[0] % args.batch_size else data.shape[
                                                                                             0] // args.batch_size
    step = 0

    # Store accuracy and loss for each batch
    batch_accs = []
    batch_losses = []

    total_inference_time = 0.0  # Track total inference time for all batches

    for features, labels in iterator:
        features = features.to(device)
        labels = labels.to(device)

        # Record start time for inference
        start_inference_time = time.time()

        # Apply adjacency perturbation
        perturbed_adj = None
        if adj_perturbation is not None and adj_matrix is not None:
            perturbed_adj = adj_perturbation(adj_matrix.to(device), apply_perturbation=True)

        predicts, _, node_weights, space_node_weights = model()

        # Record end time for inference
        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
        total_inference_time += inference_time  # Accumulate inference time
        # print(f"Train time for this batch: {inference_time:.6f} seconds")

        loss = criterion(predicts, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss detected at epoch {epoch + 1}, step {step}")
            continue

        acc = accuracy(predicts.detach(), labels.detach())[0]
        # Save batch's accuracy and loss
        batch_accs.append(acc.item())
        batch_losses.append(loss.item())

        dict_log['loss'].update(loss.item(), len(features))
        dict_log['acc'].update(acc.item(), len(features))
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"Warning: NaN or Inf in gradients at epoch {epoch + 1}, step {step}")
                    optimizer.zero_grad()
                    break
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if total_norm > args.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

        param_has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN or Inf in parameter {name}")
                param_has_nan = True

        if not param_has_nan:
            optimizer.step()

        all_steps = epoch * steps + step + 1
        if 0 == (all_steps % args.print_freq):
            lr = list(optimizer.param_groups)[0]['lr']
            now_time = time.time() - start_time
            et = str(datetime.timedelta(seconds=now_time))[:-7]
            print_information = 'id:{} time consumption:{} epoch:{}/{} lr:{} '.format(
                args.id, et, epoch + 1, args.epochs, lr)
            for key, value in dict_log.items():
                loss_info = "{}(val/avg):{:.3f}/{:.3f} ".format(key, value.val, value.avg)
                print_information = print_information + loss_info
                tensorboard.add_scalar(key, value.val, all_steps)
            print(print_information)
        step = step + 1

    # Compute min, mean, max for acc and loss
    min_acc, mean_acc, max_acc = min(batch_accs), sum(batch_accs) / len(batch_accs), max(batch_accs)
    min_loss, mean_loss, max_loss = min(batch_losses), sum(batch_losses) / len(batch_losses), max(batch_losses)

    # Compute total and average inference time
    avg_inference_time = total_inference_time / len(iterator)
    print(f"Total inference time for this epoch: {total_inference_time:.6f} seconds")
    print(f"Average inference time per batch: {avg_inference_time:.6f} seconds")

    # Return average training loss
    avg_train_loss = dict_log['loss'].avg
    print('--------------------------End training at epoch:{}--------------------------'.format(epoch + 1))
    return node_weights, space_node_weights, avg_train_loss, (min_acc, mean_acc, max_acc), (
        min_loss, mean_loss, max_loss)


def evaluate_one_epoch(epoch, iterator, data, model, device, criterion, tensorboard, args, start_time, r,
                       adj_matrix=None, adj_perturbation=None):
    print('--------------------------Start evaluating at epoch:{}--------------------------'.format(epoch + 1))
    model.to(device)
    dict_log = {'loss': AverageMeter(), 'acc': AverageMeter()}
    model.eval()
    data, data_labels = data
    step = 0
    start_time = time.time()

    all_preds = []
    all_labels = []

    total_inference_time = 0.0  # Used to track total inference time for all batches

    for features, labels in iterator:

        features = features.to(device)
        labels = labels.to(device)

        # Record start time for inference
        start_inference_time = time.time()

        with torch.no_grad():
            # Don't apply adjacency perturbation during evaluation for model determinism
            predicts, _, _, _ = model()
            loss = criterion(predicts, labels)

        # Record end time for inference
        end_inference_time = time.time()
        inference_time = end_inference_time - start_inference_time
        total_inference_time += inference_time  # Accumulate inference time
        # print(f"Inference time for this batch: {inference_time:.6f} seconds")  # Output inference time for each batch

        acc = accuracy(predicts.detach(), labels.detach())[0]
        dict_log['acc'].update(acc.item(), len(features))
        dict_log['loss'].update(loss.item(), len(features))

        all_preds.append(torch.argmax(predicts, dim=1).cpu())
        all_labels.append(labels.cpu())

    # Compute total and average inference time
    avg_inference_time = total_inference_time / len(iterator)
    print(f"Total inference time for this epoch: {total_inference_time:.6f} seconds")
    print(f"Average inference time per batch: {avg_inference_time:.6f} seconds")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    kappa = cohen_kappa_score(all_labels, all_preds)

    end_time = time.time()
    now_time = time.time() - start_time
    et = str(datetime.timedelta(seconds=now_time))[:-7]
    print_information = 'time consumption:{} epoch:{}/{} '.format(et, epoch + 1, args.epochs, len(data))

    for key, value in dict_log.items():
        loss_info = "{}(avg):{:.3f} ".format(key, value.avg)
        print_information = print_information + loss_info
        tensorboard.add_scalar(key, value.val, epoch)

    duration_time = ' ' + str(end_time - start_time)
    print(print_information + duration_time)
    print_information += f"Kappa: {kappa:.3f} "
    print('--------------------------Ending evaluating at epoch:{}--------------------------'.format(epoch + 1))
    return dict_log['acc'].avg, dict_log['loss'].avg, kappa


def train(model, device, train_loader, optimizer, epoch, scheduler=None, max_grad_norm=0.5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_count = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"Warning: NaN or Inf in input data at batch {batch_idx}")
            continue

        optimizer.zero_grad()
        output = model(data.float())

        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"Warning: NaN or Inf in model output at batch {batch_idx}")
            nan_count += 1
            if nan_count > 5:
                print("Too many NaN outputs, reducing learning rate")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                nan_count = 0
            continue

        loss = F.nll_loss(output, target)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss at batch {batch_idx}")
            continue

        loss.backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"Warning: NaN or Inf in gradients at batch {batch_idx}")
                    optimizer.zero_grad()
                    break
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        if total_norm > max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        param_has_nan = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN or Inf in parameter {name}")
                param_has_nan = True

        if not param_has_nan:
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t'
                  f'Grad Norm: {total_norm:.6f}')

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = 100. * correct / total if total > 0 else 0

    print(f'Train set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return avg_loss, accuracy
