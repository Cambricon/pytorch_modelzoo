from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    # batched_nms avoid going through jit
except ImportError:
    print("train/test without MLU!")


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output

def get_boxes(prediction, batch_size, img_size=416):
    """
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    reshape_value = torch.reshape(prediction, (-1, 1))
    num_boxes_final = reshape_value[0].item()
    all_list = [[] for _ in range(batch_size)]
    max_limit = 1
    min_limit = 0
    box_num = int((prediction.shape[1] - 64) / 7)
    for batch_idx in range(batch_size):
      index = (64 + box_num * 7) * batch_idx
      for i in range(int(reshape_value[index].item())):
        if batch_idx >=0 and batch_idx < batch_size:
            bl = max(min_limit, min(max_limit, reshape_value[index + 64 + i * 7 + 3].item()) * img_size)
            br = max(min_limit, min(max_limit, reshape_value[index + 64 + i * 7 + 4].item()) * img_size)
            bt = max(min_limit, min(max_limit, reshape_value[index + 64 + i * 7 + 5].item()) * img_size)
            bb = max(min_limit, min(max_limit, reshape_value[index + 64 + i * 7 + 6].item()) * img_size)

            if bt - bl > 0 and bb -br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[index + 64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[index + 64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[index + 64 + i * 7 + 1].item())

    output = [torch.FloatTensor(all_list[i]).reshape(-1, 7) for i in range(batch_size)]
    return output

def build_targets_train(model, targets):
    # targets = [image, class, x, y, w, h]
    nt = len(targets)
    tcls, tbox, indices, av = [], [], [], []
    multi_mlu = type(model) in (nn.parallel.DataParallel,
                                nn.parallel.DistributedDataParallel)
    reject, use_all_anchors = True, True
    for i in model.module.yolo_layers if multi_mlu else model.yolo_layers:
        # get number of grid points and anchor vec for this yolo layer
        if multi_mlu:
            ng, anchor_vec = model.module.module_list[i].ng, model.module.module_list[i].anchor_vec
        else:
            ng, anchor_vec = model.module_list[i].ng, model.module_list[i].anchor_vec
        # iou of targets-anchors
        t, a = targets, []
        gwh = t[:, 4:6] * ng
        if nt:
            iou = wh_iou(anchor_vec, gwh)
            if use_all_anchors:
                na = len(anchor_vec)  # number of anchors
                a = torch.arange(na, device=targets.device).view((-1, 1)).repeat([1, nt]).view(-1)
                t = targets.repeat([na, 1])
                gwh = gwh.repeat([na, 1])
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor
            # reject anchors below iou_thres (OPTIONAL, increases P, lowers R)
            if reject:
                j = iou.view(-1) > model.hyp['iou_t']  # iou threshold hyperparameter
                t, a, gwh = t[j], a[j], gwh[j]
        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * ng  # grid x, y
        gi, gj = gxy.long().t()  # grid x, y indices
        indices.append((b, a, gj, gi))
        # Box
        gxy = gxy - gxy.floor()  # xy
        tbox.append(torch.cat((gxy, gwh), 1))  # xywh (grids)
        av.append(anchor_vec[a])  # anchor vec
        # Class
        tcls.append(c)
        if c.shape[0]:  # if any targets
            #assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
            assert c.max() < 80, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
            #                               model.nc, model.nc - 1, c.max())
                                           80, 79, c.max())
    return tcls, tbox, indices, av

def build_targets_test(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

def mlu_float_tensor(p):
    return torch.tensor(p).float().to(ct.mlu_device())

def compute_loss(p, targets, model, giou_flag=True):  # predictions, targets, model
    if p[0].is_cuda:
        ft = torch.cuda.FloatTensor
    elif ct.is_mlu_tensor(p[0]):
        ft = mlu_float_tensor
    else:
        ft = torch.Tensor

    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets_train(model, targets)
    h = model.hyp  # hyperparameters
    arc = model.arc  # # (default, uCE, uBCE) detection architectures
    red = 'sum'  # Loss reduction (sum or mean)
    model.nc = 80

    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)
    BCE = nn.BCEWithLogitsLoss(reduction=red)
    CE = nn.CrossEntropyLoss(reduction=red)  # weight=model.class_weights

    if 'F' in arc:  # add focal loss
        g = h['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)
    # Compute losses
    np, ng = 0, 0  # number grid points, targets
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0])  # target obj
        np += tobj.numel()

        # Compute losses
        nb = len(b)
        if nb:  # number of targets
            ng += nb
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
            # ps[:, 2:4] = torch.sigmoid(ps[:, 2:4])  # wh power loss (uncomment)
            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
            tobj[b, a, gj, gi] = giou.detach().type(tobj.dtype) if giou_flag else 1.0

            if 'default' in arc and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(ps[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(ps[:, 5:], t)  # BCE
                # Instance-class weighting (use with reduction='none')
                # nt = t.sum(0) + 1  # number of targets per class
                # lcls += (BCEcls(ps[:, 5:], t) / nt).mean() * nt.mean()  # v1
                # lcls += (BCEcls(ps[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        if 'default' in arc:  # separate obj and cls
            lobj += BCEobj(pi[..., 4], tobj)  # obj loss
        elif 'BCE' in arc:  # unified BCE (80 classes)
            t = torch.zeros_like(pi[..., 5:])  # targets
            if nb:
                t[b, a, gj, gi, tcls[i]] = 1.0
            lobj += BCE(pi[..., 5:], t)
        elif 'CE' in arc:  # unified CE (1 background + 80 classes)
            t = torch.zeros_like(pi[..., 0], dtype=torch.long)  # targets
            if nb:
                t[b, a, gj, gi] = tcls[i] + 1
            lcls += CE(pi[..., 4:].view(-1, model.nc + 1), t.view(-1))
    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        lobj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if ng:
            lcls *= 3 / ng / model.nc
            lbox *= 3 / ng
    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()

