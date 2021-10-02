import torch.autograd
import cv2
from DataManagement.data_management import *
from ModelManagement.PytorchModel.DeepLab_V3_Plus import *
from UtilityManagement.AverageMeter import *
from ModelManagement.evaluator import Evaluator
import time

save_datalist()

learning_rate = cf.network_info['learning_rate']
gpu_check = is_gpu_avaliable()
devices = torch.device("cuda") if gpu_check else torch.device("cpu")

eval = Evaluator(cf.NUM_CLASSES)

trainingset = GTA5Dataset(cf.paths['train_dataset_file'])
data_loader = get_loader(trainingset, batch_size=cf.network_info['batch_size'], shuffle=True, num_worker=cf.network_info['num_worker'])

validationset = GTA5Dataset(cf.paths['valid_dataset_file'])
valid_loader = get_loader(validationset, batch_size=int(cf.network_info['batch_size']/2), shuffle=False, num_worker=cf.network_info['num_worker'])

# model test code
model = DeepLabV3Plus(cf.NUM_CLASSES).to(devices)

criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

optimizer = set_SGD(model, learning_rate=learning_rate)

best_pred = 0.0

pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    start_epoch = checkpoint['epoch']
    load_weight_parameter(model, checkpoint['state_dict'])
    load_weight_parameter(optimizer, checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']
else:
    print("No Pretrained Model")
    start_epoch = 0

for epoch in range(start_epoch, cf.network_info['epochs']):

    # Learning Rate 조절하기
    lr = learning_rate * (0.1 ** (epoch // 10))  # ResNet Lerarning Rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i_batch, sample_bathced in enumerate(data_loader):
        data_time.update(time.time() - end)

        source = sample_bathced['source']
        target = sample_bathced['target'].squeeze()
        target *= 255.0
        target = target.type(torch.long)
        if gpu_check:
            source = source.to(devices)
            target = target.to(devices)

        output = model(source)
        loss = criterion(output, target)

        losses.update(loss.item(), source.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch+1, i_batch, len(data_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

    valid_batch_time = AverageMeter()
    valid_data_time = AverageMeter()
    valid_losses = AverageMeter()

    model.eval()
    eval.reset()
    source = None
    output = None

    end = time.time()
    for i_batch, sample_bathced in enumerate(valid_loader):
        data_time.update(time.time() - end)

        source = sample_bathced['source']
        target = sample_bathced['target'].squeeze()

        target *= 255.0
        target = target.type(torch.long)
        if gpu_check:
            source = source.to(devices)
            target = target.to(devices)

        output = model(source)
        loss = criterion(output, target)

        losses.update(loss.item(), source.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        eval.add_batch(target, pred)

        if i_batch % cf.network_info['freq_print'] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(valid_loader),
                                                        batch_time=batch_time, data_time=data_time, loss=losses))

    # Input Image, Predict Image Show
    img = source.cpu().numpy()
    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    segmap = np.array(pred[0]).astype(np.uint8)
    segmap = decode_segmap(segmap)
    img = np.transpose(img[0], axes=[1, 2, 0])
    ann = target
    ann = np.array(ann[0]).astype(np.uint8)
    ann = decode_segmap(ann)
    plt.figure()
    plt.subplot(311)
    plt.imshow(img)
    plt.subplot(312)
    plt.imshow(segmap)
    plt.subplot(313)
    plt.imshow(ann)
    plt.savefig('Result_' + str(epoch) + '.png')

    Acc = eval.Pixel_Accuracy()
    Acc_class = eval.Pixel_Accuracy_Class()
    mIoU = eval.Mean_Intersection_over_Union()
    FWIoU = eval.Frequency_Weighted_Intersection_over_Union()

    print("* Validation --> Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))

    new_pred = mIoU
    if new_pred  > best_pred:
        best_pred = new_pred

    save_checkpoint({
        'epoch': epoch + 1,
        'arch' : model.get_name(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_pred': best_pred}, False, os.path.join(pretrained_path, model.get_name()),'pth')


print("Train Finished!!")
