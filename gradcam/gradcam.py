
from skimage.filters.rank.generic import gradient
import torch
import torch.nn.functional as F
from torch import nn

from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_my_layer


class GradCAM(object):
    def __init__(self, model_dict):
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch'] 

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
           
        if layer_name == 'relu1':
            target_layer = self.model_arch.relu1
        elif layer_name == 'relu2':
            target_layer = self.model_arch.relu2
        elif layer_name == 'relu3':
            target_layer = self.model_arch.relu3
        elif layer_name == 'relu4':
            target_layer = self.model_arch.relu4        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)


    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        saliency_map = gradients
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)

        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def first_layer_hook_fn(module, grad_in, grad_out):
            self.image_reconstruction = grad_in[0]
        

        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop() 
            grad[grad > 0] = 1 
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad
            return (new_grad_in,)

        modules = list(self.model.named_children())

        for name, module in modules:
            if isinstance(module, nn.SELU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class = None):
        logit = self.model(input_image.requires_grad_())
        if target_class is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, target_class].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=True)
        result = self.image_reconstruction.data[0]
        return result.cpu().numpy()
