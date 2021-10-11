import torch.autograd
import torch.backends.cudnn as cudnn
import cv2
from DataManagement.data_management import *
from ModelManagement.PytorchModel.SalsaNext import *
from ModelManagement.AverageMeter import *
from ModelManagement.Lovsaz_Softmax import Lovasz_softmax
from ModelManagement.warmupLR import warmupLR
from ModelManagement.Evaluator import Evaluator
import time
import datetime


# Load Parsing File
model_info = yaml.safe_load(open("UtilityManagement/" + cf.paths["model_info"], 'r'))
dataset_info = yaml.safe_load(open("UtilityManagement/" + cf.paths["dataset_info"], 'r'))

# Load GPU
if is_gpu_avaliable():
    devices = torch.device("cuda")
    print("I Use GPU --> Device : CUDA")
else:
    devices = torch.device("cpu")
    print("I Use CPU --> Device : CPU")

# Load Dataset
trainingset = SemanticKitti(model_info, dataset_info, True, 0)
data_loader = get_loader(trainingset, model_info["train"]["batch_size"], shuffle=True, num_worker=model_info["train"]["workers"])
print("Semantic-KITTI Training Dataset Load Success --> DataSize : {}, Sequences : {}".format(len(trainingset), trainingset.get_sequence()))

validationset = SemanticKitti(model_info, dataset_info, True, 1)
valid_loader = get_loader(trainingset, int(model_info["train"]["batch_size"] / model_info["train"]["batch_size"]), shuffle=False, num_worker=model_info["train"]["workers"])
print("Semantic-KITTI Validation Dataset Load Success --> DataSize : {}, Sequences : {}".format(len(validationset), validationset.get_sequence()))

# Load Loss Func, Optimizer
# 원 코드에서는 Epsilon을 사용하였지만, 현 코드에서는 Epsilon을 사용하지 않고 훈련을 진행할 예정이다.
# epsilon_w = model_info["train"]["epsilon_w"]
content = torch.zeros(trainingset.get_num_classes(), dtype=torch.float)
for cl, freq in dataset_info["content"].items():
    x_cl = trainingset.get_xentropy_map(cl)
    content[x_cl] += freq
loss_w = 1 / content                # loss_w = 1 / ( content + epsilon_w )
for x_cl, weight in enumerate(loss_w):
    if dataset_info["learning_ignore"][x_cl]:
        loss_w[x_cl] = 0
print("Loss Weight from Imbalanced Class : ", loss_w.data)

# Load Model
with torch.no_grad():
    model = salsanext(trainingset.get_num_classes()).to(devices)
    get_summary(model, devices)

if is_gpu_avaliable() and get_gpu_device_count() > 0:
    cudnn.benchmark = True
    cudnn.fastest = True

# Load Loss Function and Optimizer
criterion = nll_loss(loss_w).to(devices)
ls = Lovasz_softmax(ignore=0).to(devices)

optimizer = set_SGD(model=model, learning_rate=model_info["train"]["lr"], momentum=model_info["train"]["momentum"], weight_decay=model_info["train"]["w_decay"])

# Load Learning Rate Scheduler
steps_per_epoch = len(data_loader)
up_steps = int(model_info["train"]["wup_epochs"] * steps_per_epoch)
final_decay = model_info["train"]["lr_decay"] ** (1 / steps_per_epoch)
scheduler = warmupLR(optimizer=optimizer, lr=model_info["train"]["lr"], warmup_steps=up_steps, momentum=model_info["train"]["momentum"], decay=final_decay)

info = {"train_loss": 0,
        "train_acc": 0,
        "train_iou": 0,
        "valid_loss": 0,
        "valid_acc": 0,
        "valid_iou": 0,
        "best_train_iou": 0,
        "best_val_iou": 0}

# Log File Check
pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    torch.nn.Module.dump_patches = True         # Pytorch 버전이 안맞아도 작동이 될 수 있게끔 설정하는 구문
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    start_epoch = checkpoint['epoch']
    info = checkpoint["info"]
    load_weight_parameter(model, checkpoint['state_dict'])
    load_weight_parameter(optimizer, checkpoint['optimizer'])
    load_weight_parameter(scheduler, checkpoint["scheduler"])
    best_pred = checkpoint['best_pred']
else:
    print("No Pretrained Model")
    start_epoch = 0

batch_time_t = AverageMeter()
data_time_t = AverageMeter()
batch_time_e = AverageMeter()

ignore_class = []
for i, w in enumerate(loss_w):
    if w < 1e-10:
        ignore_class.append(i)
evaluator = Evaluator(trainingset.get_num_classes(), devices, ignore_class)

for epoch in range(start_epoch, model_info["train"]["max_epochs"]):

    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()

    if is_gpu_avaliable():
        torch.cuda.empty_cache()        # 사용하지 않으면서 캐시된 메모리들을 해제해준다.

    model.train()

    end = time.time()
    for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(data_loader):

        data_time_t.update(time.time() - end)

        if is_gpu_avaliable():
            in_vol = in_vol.to(devices)
            proj_labels = proj_labels.to(devices)

        output = model(in_vol)
        loss = criterion(torch.log(output.clamp(min=1e-8)), proj_labels.long()) + ls(output, proj_labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.mean()

        with torch.no_grad():
            evaluator.reset()
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)
            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()

        losses.update(loss.item(), in_vol.size(0))
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))

        batch_time_t.update(time.time() - end)      # 배치 크기만큼 불러와 학습을 마치고 BackPropagation까지 하는데 걸리는 시간
        end = time.time()

        if model_info["train"]["save_scans"]:
            # Result Color Map
            mask_np = proj_mask[0].cpu().numpy()
            depth_np = in_vol[0][0].cpu().numpy()
            pred_np = argmax[0].cpu().numpy()
            gt_np = proj_labels[0].cpu().numpy()
            out = make_log_img(depth_np, mask_np, pred_np, gt_np, trainingset.get_color)

            mask_np = proj_mask[1].cpu().numpy()
            depth_np = in_vol[1][0].cpu().numpy()
            pred_np = argmax[1].cpu().numpy()
            gt_np = proj_labels[1].cpu().numpy()
            out2 = make_log_img(depth_np, mask_np, pred_np, gt_np, trainingset.get_color)

            out = np.concatenate([out, out2], axis=0)
            cv2.imwrite(cf.paths["save_path"] + "/" + str(i) + ".png", out)

        # ------------------------------------------

        for g in optimizer.param_groups:
            lr = g["lr"]

        estimate = int((data_time_t.avg + batch_time_t.avg) * \
                       (len(data_loader) * model_info['train']['max_epochs'] - (
                               i + 1 + epoch * len(data_loader)))) + \
                   int(batch_time_e.avg * len(valid_loader) * (
                           model_info['train']['max_epochs'] - (epoch)))

        if i % model_info["train"]["report_batch"] == 0:
            print('Lr: {lr:.3e} | '
                  'Epoch: [{0}][{1}/{2}] | '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                  'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                  'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]'.format(
                epoch, i, len(data_loader), batch_time=batch_time_t,
                data_time=data_time_t, loss=losses, acc=acc, iou=iou, lr=lr,
                estim=str(datetime.timedelta(seconds=estimate))))

        scheduler.step()

    # Epoch Finish
    info["train_loss"] = loss.avg
    info["train_acc"] = acc.avg
    info["train_iou"] = iou.avg

    state = {'epoch': epoch, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'info': info,
             'scheduler': scheduler.state_dict()}

    torch.save(state, os.path.join(pretrained_path, model.get_name()) + '.pth')

    if info['train_iou'] > info['best_train_iou']:
        print("Best mean iou in training set so far, save model!")
        info['best_train_iou'] = info['train_iou']
        state = {'epoch': epoch, 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'info': info,
                 'scheduler': scheduler.state_dict()}
        torch.save(state, os.path.join(pretrained_path, model.get_name()) + '_train_best.pth')

# Validation Process

    if epoch % model_info["train"]["report_epoch"] == 0:
        print("*" * 20)

        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        model.eval()
        evaluator.reset()

        if is_gpu_avaliable():
            torch.cuda.empty_cache()        # 사용하지 않으면서 캐시된 메모리들을 해제해준다.

        with torch.no_grad():
            end = time.time()
            for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(valid_loader):
                if is_gpu_avaliable():
                    in_vol = in_vol.to(devices)
                    proj_labels = proj_labels.to(devices)

                output = model(in_vol)
                log_out = torch.log(output.clamp(min=1e-8))
                jacc = ls(output, proj_labels)
                wce = criterion(log_out, proj_labels)
                loss = wce + jacc

                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)

                losses.update(loss.mean().item(), in_vol.size(0))
                jaccs.update(jacc.mean().item(), in_vol.size(0))
                wces.update(wce.mean().item(), in_vol.size(0))

                batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = evaluator.getacc()
            jaccard, class_jaccard = evaluator.getIoU()
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))

            print('Validation set:\n'
                  'Time avg per batch {batch_time.avg:.3f}\n'
                  'Loss avg {loss.avg:.4f}\n'
                  'Jaccard avg {jac.avg:.4f}\n'
                  'WCE avg {wces.avg:.4f}\n'
                  'Acc avg {acc.avg:.3f}\n'
                  'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time_e,
                                                 loss=losses,
                                                 jac=jaccs,
                                                 wces=wces,
                                                 acc=acc, iou=iou))

            for i, jacc in enumerate(class_jaccard):
                print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                    i=i, class_str=validationset.get_xentropy_class_string(i), jacc=jacc))
                info["valid_classes/" + validationset.get_xentropy_class_string(i)] = jacc

        # update info
        info["valid_loss"] = losses.avg
        info["valid_acc"] = acc.avg
        info["valid_iou"] = iou.avg

        if info['valid_iou'] > info['best_val_iou']:
            print("Best mean iou in validation so far, save model!")
            print("*" * 20)
            info['best_val_iou'] = info['valid_iou']

            # save the weights!
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'info': info,
                     'scheduler': scheduler.state_dict()
                     }
            torch.save(state, os.path.join(pretrained_path, model.get_name()) + "_valid_best.pth")

        print("*" * 80)


print("Train Finished!!")
