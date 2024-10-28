import math
import torch

from yolov6.utils.downloads import attempt_download_from_hub, attempt_download
from yolov6.utils.events import LOGGER
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from yolov6.utils.coco_classes import COCO_CLASSES
        
def check_img_size(img_size, s=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

def model_switch(model):
    ''' Model switch to deploy status '''
    from yolov6.layers.common import RepVGGBlock
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    LOGGER.info("Switch model to deploy modality.")
    
class YOLOV6:
    def __init__(
        self,
        weights = 'weights/yolov6s.pt',
        device = 'cpu',
        hf_model = False,
    ):

        self.__dict__.update(locals())
        self.device = device
        self.half = False
        # Load model
        if hf_model:
            self.weights = attempt_download_from_hub(weights, hf_token=None)
        else:
            self.weights = attempt_download(weights)
        model = self.load_model()
        self.stride = model.stride

        # Model Parameters
        self.conf = 0.25
        self.iou = 0.45
        self.classes = None
        self.annotate = False
        self.agnostic_nms = False
        self.max_det = 1000


    def load_model(self):
        # Init model
        model = DetectBackend(self.weights, device=self.device)

        # Switch model to deploy status
        model_switch(model.model)

        # Half precision
        if self.half & (self.device != 'cpu'):
            model.model.half()
        else:
            model.model.float()
            self.half = False

        self.model = model
        return model


    def predict(self,image,img_size=640):
        img_size = check_img_size(img_size, s=self.stride)
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, *img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        img, img_src = Inferer.precess_image(image, img_size, self.stride, self.half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
        pred_results = self.model(img)
        det = non_max_suppression(pred_results, self.conf, self.iou, classes=self.classes, agnostic=self.agnostic_nms, max_det=self.max_det)[0]
        det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
        if self.annotate:
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)  # integer class
                label = f'{COCO_CLASSES[class_num]} {conf:.2f}'
                Inferer.plot_box_and_label(image, max(round(sum(image.shape) / 2 * 0.001), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
        return det