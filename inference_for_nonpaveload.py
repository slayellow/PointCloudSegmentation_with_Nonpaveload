import torch.autograd
import torch.backends.cudnn as cudnn
import cv2
from DataManagement.data_management_changwon import *
from ModelManagement.PytorchModel.SalsaNext import *
from ModelManagement.AverageMeter import *
from ModelManagement.Lovsaz_Softmax import Lovasz_softmax
from ModelManagement.warmupLR import warmupLR
from ModelManagement.Evaluator import Evaluator
from ModelManagement.KNN import KNN
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
testset = Changwon(model_info, dataset_info)
data_loader = get_loader(testset, 1, shuffle=False, num_worker=model_info["test"]["workers"])
print("Changwon Test Dataset Load Success --> DataSize : {}, Sequences : {}".format(len(testset), testset.get_sequence()))

# Load Model
with torch.no_grad():
    torch.nn.Module.dump_patches = True
    model = salsanext(testset.get_num_classes()).to(devices)
    get_summary(model, devices)

if is_gpu_avaliable() and get_gpu_device_count() > 0:
    cudnn.benchmark = True
    cudnn.fastest = True

# use knn post processing?
# KNN 분석 필ㄲㄷ녀ㅣㅅ
post = None
if model_info["post"]["KNN"]["use"]:
    post = KNN(model_info["post"]["KNN"]["params"],
                  testset.get_num_classes())

# Log File Check
pretrained_path = cf.paths['pretrained_path']
if os.path.isfile(os.path.join(pretrained_path, model.get_name() + '.pth')):
    torch.nn.Module.dump_patches = True         # Pytorch 버전이 안맞아도 작동이 될 수 있게끔 설정하는 구문
    print("Pretrained Model Open : ", model.get_name() + ".pth")
    checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
    load_weight_parameter(model, checkpoint['state_dict'])
else:
    print("No Pretrained Model")
    start_epoch = 0

cnn = []
knn = []
model.eval()

total_time=0
total_frames=0
# empty the cache to infer in high res
if is_gpu_avaliable():
  torch.cuda.empty_cache()

with torch.no_grad():
    end = time.time()

    for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(data_loader):
        # first cut to rela size (batch size one allows it)

        p_x = p_x[0, :npoints[0]]
        p_y = p_y[0, :npoints[0]]
        proj_range = proj_range[0, :npoints[0]]
        unproj_range = unproj_range[0, :npoints[0]]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if is_gpu_avaliable():
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        proj_output = model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("Network seq", path_seq, "scan", path_name,
              "in", res, "sec")
        end = time.time()
        cnn.append(res)

        if post:
            # knn postproc
            unproj_argmax = post(proj_range,
                                      unproj_range,
                                      proj_argmax,
                                      p_x,
                                      p_y)
        else:
            # put in original pointcloud using indexes
            unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = time.time() - end
        print("KNN Infered seq", path_seq, "scan", path_name,
              "in", res, "sec")
        knn.append(res)
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = testset.get_original(pred_np)

        # save scan
        path = os.path.join(cf.paths["changwon_save_path"], path_name)
        pred_np.tofile(path)