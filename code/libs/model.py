import math
import torch
import torchvision


from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss

INF = 100000000


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 2.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=2, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        logits = []
        for l, feature in enumerate(x):
            cls_tower = self.conv(feature)
            logits.append(self.cls_logits(cls_tower))
        
        return logits


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 2.
    """

    def __init__(self, in_channels, num_convs=2):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        bboxreg=[]
        centerness=[]
        
        for l, feature in enumerate(x):
            bbox_tower = self.conv(feature)
            centerness.append(self.bbox_ctrness(bbox_tower))
            bboxreg.append(self.bbox_reg(bbox_tower))
        
        return bboxreg, centerness


class FCOS(nn.Module):
    """
    Implementation of (simplified) Fully Convolutional One-Stage object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet18 is supported
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone == "ResNet18"
        self.backbone_name = backbone
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network (resnet18)
        self.backbone = create_feature_extractor(
            resnet18(weights=ResNet18_Weights.DEFAULT), return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
            else:
                module.train(mode)
        return self

    """
    The behavior of the forward function changes depending if the model is
    in training or evaluation mode.

    During training, the model expects both the input tensors
    (list of tensors within the range of [0, 1]), as well as a targets
    (list of dictionary), containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)
        
        #num_points = [center.size(0) for center in points]
        #print('num_points',num_points)
        #print('num_images',cls_logits[0].size(0))

        # training / inference
        if self.training:
            # training: generate G3T labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        all_level_points = []
        #points ([],[],[])each WxHx2
        for i in range (len(points)):
            all_level_points.append(points[i].view(-1,2))

        #print(points[0].shape,points[1].shape,points[2].shape)
        #print(all_level_points[0].shape,all_level_points[1].shape,all_level_points[2].shape)
        #all_level_points ([],[],[])each (W*H)x2

        #labels, bbox_targets = self.get_targets(points,)

        num_imgs = cls_logits[0].size(0)
        labels, bbox_targets = self.get_targets(all_level_points, targets, strides, reg_range)

        #print(len(bbox_targets),bbox_targets[0].shape,bbox_targets[1].shape,bbox_targets[2].shape)
        #print(len(cls_logits))
        #print(cls_logits[0].shape)

        #make shape as [(h*w*n, 20)*3]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_logits
        ]
        #print(flatten_cls_scores[0].shape)
        # make shape as flatten_bbox_preds = [(h*w*n, 4)*3]
        #print(reg_outputs[1].shape)
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in reg_outputs
        ]
        #print(len(flatten_bbox_preds),flatten_bbox_preds[1].shape)
        #make shape as flatten_bbox_preds = [(h*w*n)*3]
        #print(ctr_logits[2].shape)
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in ctr_logits
        ]
        #print(len(flatten_centerness),flatten_centerness[2].shape)

        # flatten_cls_scores = [sum(h*w*n) x 20] for all images
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        #print(flatten_cls_scores.shape)
        # flatten_bbox_preds = [sum(h*w*n) x 4]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        #print(flatten_bbox_preds.shape)
        # flatten_centerness = [sum(h*w*n) x 1]
        flatten_centerness = torch.cat(flatten_centerness)
        #print(flatten_centerness.shape)
        # flatten_labels = [sum(h*w*n) x 1]
        #after映射
        flatten_labels = torch.cat(labels)
        #print(flatten_labels.shape)
        # flatten_bbox_targets = [sum(h*w*n) x 4]
        #after映射
        flatten_bbox_targets = torch.cat(bbox_targets)
        #print(flatten_bbox_targets.shape)
        # flatten_points = [sum(h*w*n) x 2]
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        #print(flatten_points.shape)

        # the positive label's range
        # positive label'range fg_class_ind: [0, num_classes -1], negative label'range bg_class_ind: num_classes
        bg_class_ind = self.num_classes
        #the inds of positive labels
        #shape: (num_pos_anchors)
        pos_inds = ((flatten_labels >= 0)
            & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        #print(pos_inds.shape,pos_inds)

        num_pos = len(pos_inds)
        #print(num_pos)
        #print(flatten_cls_scores)
        if num_pos == 0:
            num_pos = num_imgs
        #print(flatten_labels)

        #make the 
        #f_cls_scores=torch.argmax(flatten_cls_scores,axis=1)
        #print(f_cls_scores)

        #for i in range(flatten_labels.shape[0]):
         #   if flatten_labels[i] == 20:
          #      flatten_labels[i] = 0
           # else:
            #    flatten_labels[i] = 1
        #print(flatten_labels)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #aa=torch.empty(flatten_cls_scores.shape[0],self.num_classes+1).to(device)
        #cls_scores_onehot = torch.zeros_like(aa)
        #print(cls_scores_onehot[0:4,:])
        #cls_scores_onehot.scatter_(1,flatten_labels,1)
        n_21 = self.num_classes + 1
        flatten_labels_one_hot = torch.nn.functional.one_hot(flatten_labels % n_21)
        #print(flatten_labels_one_hot[0:3,:],flatten_labels_one_hot.shape)
        flatten_labels_one_hot = flatten_labels_one_hot[:,:-1]

        #compute the loss for classification
        loss_cls = sigmoid_focal_loss(
            flatten_cls_scores, flatten_labels_one_hot,reduction="sum")/num_pos
        #print(loss_cls)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]

        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        #print(pos_centerness)
        #print(pos_centerness_targets)
        #print(pos_bbox_preds)

        if num_pos > 0:
            #anchor points positive
            pos_points = flatten_points[pos_inds]
            # STEP 13: 得到预测的Bbox坐标
            # 根据正样本 anchor point 所在的位置以及预测出来的四条边来将折四条边还原成Bbox的坐标格式: (x1, y1, x2, y2)
 
            pos_decoded_bbox_preds = self.bbox_decoder(
                pos_points, pos_bbox_preds)
            # 得到GT的Bbox坐标
            pos_decoded_target_preds = self.bbox_decoder(
                pos_points, pos_bbox_targets)

            loss_bbox = giou_loss(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                reduction="sum"
            )/num_pos
            #print(loss_bbox)
            Loss_c = nn.BCEWithLogitsLoss(reduction='sum')
            #print(pos_centerness.shape,pos_centerness_targets.shape)
            loss_centerness= Loss_c(
                pos_centerness,
                pos_centerness_targets,
            )/num_pos
            #print(loss_centerness)

        else:
            reg_loss = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
        #loss_cls = loss_cls.sum()
        #print('cls',loss_cls)
        #loss_bbox = loss_bbox.sum()
        #print('bbox',loss_bbox)
        final_loss = loss_cls+loss_bbox+loss_centerness
        #final_loss=loss_cls.clone()+loss_centerness
        #final_loss[pos_inds]=loss_cls[pos_inds]+loss_bbox

        losses={
            "cls_loss":loss_cls,
            "reg_loss":loss_bbox,
            "ctr_loss":loss_centerness,
            "final_loss":final_loss

        }

        return losses

    def bbox_decoder(self,pos_points, pos_bbox_preds):
        l=pos_bbox_preds[:,0]
        r=pos_bbox_preds[:,2]
        t=pos_bbox_preds[:,1]
        b=pos_bbox_preds[:,3]
        x=pos_points[:,0]
        y=pos_points[:,1]

        x1 = - l + x
        x2 = r + x
        y1 = - t + y
        y2 = b + y
        bbox_encoded = torch.stack((x1,y1,x2,y2),-1)
        return bbox_encoded


    def centerness_target(self, pos_bbox_targets):
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)





    def get_targets(self, points, targets, strides, reg_range):
        num_levels = len(points)

        gt_labels = []
        gt_bboxes = []
        for target in targets:
            gt_labels.append(target["labels"])
            gt_bboxes.append(target["boxes"])

        #print('num',len(gt_bboxes))
        #for i in range(4):
            #print(gt_labels[i].shape)
        #print(gt_bboxes[0].shape,gt_bboxes[1].shape,gt_bboxes[2].shape,gt_bboxes[3].shape)

        num_images = len(gt_labels)


        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(reg_range[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        concat_regress_ranges = torch.cat(expanded_object_sizes_of_interest, dim=0)
        #make the concat_regress_ranges and concat_points the same size
        num_points_per_level = [center.size(0) for center in points]

        concat_points = torch.cat(points, dim=0)
        #print(concat_regress_ranges.shape,concat_points.shape)

        #get labels and bbox_targets of each image for all anchor points
        labels_list = []
        bbox_targets_list = []
        for i in range(num_images):
            l, b = self.get_target_single(concat_points, gt_bboxes[i],gt_labels[i], strides, concat_regress_ranges, num_points_per_level)
            labels_list.append(l)
            bbox_targets_list.append(b)
        #print(labels_list[0].shape,bbox_targets_list[1].shape)

        #labels_list[0][0].shape+labels_list[0][1].shape+labels_list[0][2].shape=num_points_all
        labels_list = [labels.split(num_points_per_level, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points_per_level, 0) for bbox_targets in bbox_targets_list]
        #print(labels_list[0][0].shape,labels_list[0][1].shape)

        #the concat of labels of all images of one level
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            bbox_targets = bbox_targets / strides[i]
            #divide by stride
            concat_lvl_bbox_targets.append(bbox_targets)

        #print(len(concat_lvl_bbox_targets),concat_lvl_bbox_targets[0].shape,concat_lvl_bbox_targets[1].shape,concat_lvl_bbox_targets[2].shape)
        

        return concat_lvl_labels, concat_lvl_bbox_targets

    def get_target_single(self, points, gt_bboxes, gt_labels, strides, regress_ranges, num_points_per_lvl):
        #Compute regression and classification targets for a single image.
        num_points = points.shape[0]
        num_gts = gt_labels.shape[0]
        #print('11',num_points,num_gts)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)#shape: (num_anchor_point, num_gt)
        #print(areas.shape)

        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)#shape:(num_anchor_point, num_gt, 2)
        #print(regress_ranges.shape)

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)#shape:(num_anchor_point, num_gt, 4) 
        #print(gt_bboxes.shape)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)#shape: (num_anchor_point, num_gt)
        ys = ys[:, None].expand(num_points, num_gts)#shape: (num_anchor_point, num_gt)
        #print(xs.shape)

        # shape: (num_anchor_point, num_gt) 
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        #print('left',left.shape)

        #Center-Sampling
        bbox_targets = torch.stack((left, top, right, bottom), -1)# shape:(num_anchor_point, num_gt, 4)
        #print('b_t',bbox_targets.shape)

        radius = self.center_sampling_radius
        center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2# shape:(num_anchor_point, num_gt)
        center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
        #print('center_xs',center_ys.shape)

        center_gts = torch.zeros_like(gt_bboxes)#shape: (num_anchor_point, num_gt, 4)

        #print('center_gts',center_gts.shape)

        stride = center_xs.new_zeros(center_xs.shape)#shape: (num_anchor_point, num_gt)
        #print('stride',stride.shape)

        #only treat achor points within the stride*radius of gt_bbox as positive
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = strides[lvl_idx] * radius
            lvl_begin = lvl_end

        x_mins = center_xs - stride
        y_mins = center_ys - stride
        x_maxs = center_xs + stride
        y_maxs = center_ys + stride
        center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
        center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],y_mins, gt_bboxes[..., 1])
        center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],gt_bboxes[..., 2], x_maxs)
        center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],gt_bboxes[..., 3], y_maxs)
        #shape:(num_anchor_point, num_gt, 4)


        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack((cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        #shape:(num_anchor_point, num_gt, 4)

        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0#shape: (num_anchor_point, num_gt)
        #new l,t,b,r within the bbox; True/False
        #print(inside_gt_bbox_mask.shape)


        max_regress_distance = bbox_targets.max(-1)[0]#shape: (num_anchor_point, num_gt)
        #print(max_regress_distance.shape)

        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))
        #maximum of ltbr within the regress_range
        #print(inside_regress_range.shape)

        #set the unpaired area as INF, then if one anchor point has 
        #several paired area, select the bbox with min-area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF

        min_area, min_area_inds = areas.min(dim=1)
        #areas: num_gt kinds of elements
        #print(min_area.shape,min_area[min_area != INF])
        #print("min_area_inds",min_area_inds.shape)
        #print("range(num_points)",range(num_points))
        #print(gt_labels)
        labels=gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        #print(labels.shape)

        #get bbox_targets of anchor points in an image, divide stride respectively
        #stride = stride[:, :, None].expand(num_points, num_gts, 4)
        #print(bbox_targets.shape,stride.shape)
        #bbox_targets = bbox_targets/stride
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        #print("bbox_targets_ind",bbox_targets)
        #print(bbox_targets.shape)



        return labels, bbox_targets


    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) deocde the boxes
        (3) only keep boxes with scores larger than self.score_thresh
    (b) Combine all object candidates across levels and keep the top K (self.topk_candidates)
    (c) Remove boxes outside of the image boundaries (due to padding)
    (d) Run non-maximum suppression to remove any duplicated boxes
    (e) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from two different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels needed to be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        return detections
