import torch
import torch.nn as nn
from torch.autograd import Function
import copy
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import numpy as np
from math import log

import torch
import torch.nn as nn

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmGRL(nn.Module):
    """Gradient Reverse Layer with warm start
        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 1000
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = True):
        super(WarmGRL, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.coeff_log = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        self.coeff_log = coeff
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increment the iteration counter by 1"""
        self.iter_num += 1

    def log_status(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, (str, int, float))}


# Inspired by https://arxiv.org/abs/1606.00709
class ConjugateDualFunction:

    def __init__(self, divergence_name, gamma=4):
        self.f_div_name = divergence_name
        self.gamma = gamma

    def T(self, v):
        """Transformation T(v)"""
        if self.f_div_name == "tv":
            return 0.5 * torch.tanh(v)
        elif self.f_div_name == "kl":
            return v
        elif self.f_div_name == "jeff":
            return v
        elif self.f_div_name == "klrev":
            return v
        elif self.f_div_name == "chi":
            return v
        elif self.f_div_name == "klabs":
            return v
        elif self.f_div_name == "chiabs":
            return v
        elif self.f_div_name == "optkl":
            return v
        elif self.f_div_name == "optchi":
            return v
        elif self.f_div_name == "pearson":
            return v
        elif self.f_div_name == "neyman":
            return 1.0 - torch.exp(v)
        elif self.f_div_name == "hellinger":
            return 1.0 - torch.exp(v)
        elif self.f_div_name == "jensen":
            return log(2.0) - F.softplus(-v)
        elif self.f_div_name == "gammajensen":
            return -self.gamma * log(self.gamma) - F.softplus(-v)
        else:
            raise ValueError("Unknown divergence.")

    def fstarT(self, v):
        """Conjugate f* evaluated at T(v)"""
        if self.f_div_name == "tv":
            return 0.5 * torch.tanh(v)
        elif self.f_div_name == "kl":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "jeff":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "klabs":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "klrev":
            return -1.0 - v
        elif self.f_div_name == "optkl":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "chi":
            return 0.25 * v * v + v
        elif self.f_div_name == "chiabs":
            return 0.25 * v * v + v
        elif self.f_div_name == "pearson":
            return 0.25 * v * v + v
        elif self.f_div_name == "neyman":
            return 2.0 - 2.0 * torch.exp(0.5 * v)
        elif self.f_div_name == "hellinger":
            return torch.exp(-v) - 1.0
        elif self.f_div_name == "jensen":
            return F.softplus(v) - log(2.0)
        elif self.f_div_name == "gammajensen":
            gf = lambda v_: -self.gamma * log(self.gamma) - F.softplus(-v_)
            return -torch.log(self.gamma + 1. - self.gamma * torch.exp(gf(v))) / self.gamma
        else:
            raise ValueError("Unknown divergence.")


class fConjugateDualFunction:

    def __init__(self, divergence_name, gamma=4):
        self.f_div_name = divergence_name
        self.gamma = gamma

    def T(self, v):
        """Transformation T(v)"""
        if self.f_div_name == "tv":
            return 0.5 * torch.tanh(v)
        elif self.f_div_name == "kl":
            return v
        elif self.f_div_name == "jeff":
            return v
        elif self.f_div_name == "klrev":
            return -torch.exp(v)
        elif self.f_div_name == "pearson":
            return v
        elif self.f_div_name == "chi":
            return v
        elif self.f_div_name == "neyman":
            return 1.0 - torch.exp(v)
        elif self.f_div_name == "hellinger":
            return 1.0 - torch.exp(v)
        elif self.f_div_name == "jensen":
            return log(2.0) - F.softplus(-v)
        elif self.f_div_name == "gammajensen":
            return -self.gamma * log(self.gamma) - F.softplus(-v)
        else:
            raise ValueError("Unknown divergence.")

    def fstarT(self, v):
        """Conjugate f* evaluated at T(v)"""
        if self.f_div_name == "tv":
            return 0.5 * torch.tanh(v)
        elif self.f_div_name == "kl":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "jeff":
            return torch.exp(v - 1.0)
        elif self.f_div_name == "klrev":
            return -1.0 - v
        elif self.f_div_name == "chi":
            return 0.25 * v * v + v
        elif self.f_div_name == "pearson":
            return 0.25 * v * v + v
        elif self.f_div_name == "neyman":
            return 2.0 - 2.0 * torch.exp(0.5 * v)
        elif self.f_div_name == "hellinger":
            return torch.exp(-v) - 1.0
        elif self.f_div_name == "jensen":
            return F.softplus(v) - log(2.0)
        elif self.f_div_name == "gammajensen":
            gf = lambda v_: -self.gamma * log(self.gamma) - F.softplus(-v_)
            return -torch.log(self.gamma + 1. - self.gamma * torch.exp(gf(v))) / self.gamma
        else:
            raise ValueError("Unknown divergence.")




class fDAADLearner(nn.Module):
    def __init__(self, backbone, taskhead, taskloss, divergence, bootleneck=None, reg_coef=1, n_classes=-1,
                 aux_head=None,
                 grl_params=None,
                 learnable=True,
                 transform_type="affine",
                 init_params={"a": 1, "b": 0}):
        """
        fDAAD Learner.
        :param backbone: z=backbone(input). Thus backbone must be nn.Module. (e.g., ResNet without final FC layers) Feature extractor.
        :param taskhead: prediction = taskhead(z). Thus taskhead must be nn.Module (e.g., final FC layers of ResNet) Regressor/classifier.
        :param taskloss: Loss function used for training, e.g., nn.CrossEntropy().
        :param divergence: Name of divergence to use, e.g., 'pearson' or 'jensen'.
        :param bottleneck: (optional) Bottleneck layer after feature extractor and before classifier.
        :param reg_coef: Weight for the domain adaptation loss (fDAAD gamma coefficient).
        :param n_classes: Number of output classes; if <=1, uses a global discriminator.
        :param aux_head: (optional) If provided, used as the domain discriminator head; otherwise built from taskhead.
        :param grl_params: Dictionary of parameters for the warm-start GRL.
        """
        super(fDAADLearner, self).__init__()
        self.backbone = backbone
        self.taskhead = taskhead
        self.taskloss = taskloss
        self.bootleneck = bootleneck
        self.n_classes = n_classes
        self.reg_coeff = reg_coef
        self.auxhead = aux_head if aux_head is not None else self.build_aux_head_()

        self.fdaad_divhead = fDAADivergenceHead(divergence, self.auxhead, n_classes=self.n_classes,
                                               grl_params=grl_params,
                                               reg_coef=reg_coef,
                                               learnable=learnable,
                                               transform_type=transform_type,
                                               init_params=init_params)

    def build_aux_head_(self):
        # Duplicate taskhead to create auxiliary discriminator head
        auxhead = copy.deepcopy(self.taskhead)
        if self.n_classes == -1:
            # Create a global discriminator if no classes specified
            aux_linear = auxhead[-1]
            auxhead[-1] = nn.Sequential(
                nn.Linear(aux_linear.in_features, 1)
            )

        # Initialize parameters independently of the main task head
        auxhead.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        return auxhead

    def forward(self, x, y, src_size=-1, trg_size=-1):
        # input: (x_source, x_target), y: source (and optional target) labels
        if isinstance(x, tuple):
            src_size = x[0].shape[0]
            trg_size = x[1].shape[0]
            x = torch.cat((x[0], x[1]), dim=0)

        y_s, y_t = (y, None) if not isinstance(y, tuple) else y

        f = self.backbone(x)
        f = self.bootleneck(f) if self.bootleneck is not None else f
        net_output = self.taskhead(f)

        # Split features and outputs for source and target
        f_source = f.narrow(0, 0, src_size)
        f_tgt = f.narrow(0, src_size, trg_size)
        outputs_src = net_output.narrow(0, 0, src_size)
        outputs_tgt = net_output.narrow(0, src_size, trg_size)

        # Compute task losses
        task_loss = self.taskloss(outputs_src, y_s)
        task_loss += 0.0 if y_t is None else self.taskloss(outputs_tgt, y_t)

        # Compute adaptation loss if coefficient > 0
        fdaad_loss = 0.0
        if self.reg_coeff > 0.:
            fdaad_loss = self.fdaad_divhead(f_source, f_tgt, outputs_src, outputs_tgt)
            total_loss = task_loss + fdaad_loss
        else:
            total_loss = task_loss

        return total_loss, {
            "pred_s": outputs_src,
            "pred_t": outputs_tgt,
            "taskloss": task_loss,
            "fdaad_loss": fdaad_loss,
            "fdaad_src": self.fdaad_divhead.internal_stats["lhatsrc"],
            "fdaad_trg": self.fdaad_divhead.internal_stats["lhattrg"],
        }

    def get_reusable_model(self, pack=False):
        """
        Return parts of the model for inference.
        :param pack: If True, return a single nn.Sequential(backbone, taskhead).
        """
        if pack:
            return nn.Sequential(self.backbone, self.taskhead)
        return self.backbone, self.taskhead


class fDAADivergenceHead(nn.Module):
    # def __init__(self, divergence_name, aux_head, n_classes, grl_params=None, reg_coef=1., alpha, beta):
    def __init__(self, divergence_name, aux_head, n_classes, grl_params=None, reg_coef=1.,
            learnable=True,
            transform_type="affine",
            init_params={"a": 1, "b": 0}):

        """
        :param divergence_name: Name of divergence, e.g., 'pearson' or 'jensen'.
        :param aux_head: Auxiliary discriminator head (see paper Fig.1).
        :param n_classes: Number of classes; if <=1, uses a global discriminator.
        :param grl_params: Parameters for the warm-start gradient reversal layer.
        :param reg_coef: Regularization coefficient (default 1.0).
        """
        super(fDAADivergenceHead, self).__init__()
        self.grl = WarmGRL(auto_step=True) if grl_params is None else WarmGRL(**grl_params)
        self.aux_head = aux_head
        self.fdaad_loss = DAADLoss(divergence_name, gamma=1.0,
                                  learnable=learnable,
                                  transform_type=transform_type,
                                  init_params=init_params)
        self.internal_stats = self.fdaad_loss.internal_stats
        self.n_classes = n_classes
        self.reg_coef = reg_coef

    def forward(self, features_s, features_t, pred_src, pred_trg) -> torch.Tensor:
        """
        :param features_s: Features from source data.
        :param features_t: Features from target data.
        :param pred_src: Source predictions (logits).
        :param pred_trg: Target predictions (logits).
        :return: Adaptation loss.
        """
        f = self.grl(torch.cat((features_s, features_t), dim=0))
        src_size = features_s.shape[0]
        trg_size = features_t.shape[0]

        aux_output_f = self.aux_head(f)
        y_s_adv = aux_output_f.narrow(0, 0, src_size)
        y_t_adv = aux_output_f.narrow(0, src_size, trg_size)

        loss = self.fdaad_loss(pred_src, pred_trg, y_s_adv, y_t_adv, self.n_classes)
        self.internal_stats = self.fdaad_loss.internal_stats  # Debug info

        return self.reg_coef * loss


class TauTransform(nn.Module):
    """
    General τ-transform module supporting multiple transform types with optional learnability.

    Args:
        transform_type (str): Type of transform: "affine", "power", "exponential", or "sigmoid".
        learnable (bool): If True, parameters are trainable; otherwise fixed.
        init_params (dict): Initialization parameters.
            - affine: {"a": <initial a>, "b": <initial b>}
            - power: {"c": <initial exponent>}
            - exponential/sigmoid: {"scale": <initial scale>}
    """
    def __init__(self, transform_type="affine", learnable=False, init_params=None):
        super(TauTransform, self).__init__()
        self.transform_type = transform_type.lower()
        self.learnable = learnable

        # Set default init parameters
        if init_params is None:
            if self.transform_type == "affine":
                init_params = {"a": 1.0, "b": 0.0}
            elif self.transform_type == "power":
                init_params = {"c": 1.0}
            elif self.transform_type in ["exponential", "sigmoid"]:
                init_params = {"scale": 1.0}
            else:
                raise ValueError("Unsupported transform type.")

        # Initialize parameters or buffers
        if self.transform_type == "affine":
            if self.learnable:
                self.a = nn.Parameter(torch.tensor(init_params.get("a", 1.0), dtype=torch.float32))
                self.b = nn.Parameter(torch.tensor(init_params.get("b", 0.0), dtype=torch.float32))
            else:
                self.register_buffer("a", torch.tensor(init_params.get("a", 1.0), dtype=torch.float32))
                self.register_buffer("b", torch.tensor(init_params.get("b", 0.0), dtype=torch.float32))
        elif self.transform_type == "power":
            if self.learnable:
                self.c = nn.Parameter(torch.tensor(init_params.get("c", 1.0), dtype=torch.float32))
            else:
                self.register_buffer("c", torch.tensor(init_params.get("c", 1.0), dtype=torch.float32))
        elif self.transform_type == "exponential":
            if self.learnable:
                self.scale = nn.Parameter(torch.tensor(init_params.get("scale", 1.0), dtype=torch.float32))
            else:
                self.register_buffer("scale", torch.tensor(init_params.get("scale", 1.0), dtype=torch.float32))
        elif self.transform_type == "sigmoid":
            if self.learnable:
                self.scale = nn.Parameter(torch.tensor(init_params.get("scale", 1.0), dtype=torch.float32))
            else:
                self.register_buffer("scale", torch.tensor(init_params.get("scale", 1.0), dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported transform_type: {self.transform_type}")
    
    def forward(self, x):
        """
        Forward pass applying the selected transform.
        x: Input tensor of any shape.
        """
        if self.transform_type == "affine":
            # Affine transform: τ(x) = a * x + b
            return self.a * x + self.b
        elif self.transform_type == "power":
            # Power transform: τ(x) = |x|^c, with small eps for stability
            eps = 1e-6
            return torch.pow(torch.abs(x) + eps, self.c)
        elif self.transform_type == "exponential":
            # Exponential transform: τ(x) = exp(scale * x)
            return torch.exp(self.scale * x)
        elif self.transform_type == "sigmoid":
            # Sigmoid transform: τ(x) = sigmoid(scale * x)
            return torch.sigmoid(self.scale * x)
        else:
            raise ValueError(f"Unsupported transform_type: {self.transform_type}")



class DAADLoss(nn.Module):
    def __init__(self, divergence_name, gamma, learnable=True,
        transform_type="affine",init_params={"a": 1, "b": 0}):
        super(DAADLoss, self).__init__()

        self.lhat = None
        self.phistar = None
        self.phistar_gf = None
        self.multiplier = 1.
        self.internal_stats = {}
        self.domain_discriminator_accuracy = -1

        self.gammaw = gamma
        self.phistar_gf = lambda t: ConjugateDualFunction(divergence_name).fstarT(t)
        self.gf = lambda v: ConjugateDualFunction(divergence_name).T(v)
        self.tau_transform = TauTransform(transform_type="affine", learnable=True, init_params={"a": 1, "b": 0})

    def forward(self, y_s, y_t, y_s_adv, y_t_adv, K):

        v_s = y_s_adv
        v_t = y_t_adv

        if K > 1:
            _, prediction_s = y_s.max(dim=1)
            _, prediction_t = y_t.max(dim=1)

            # This is not used here as a loss, it just a way to pick elements.

            # picking element prediction_s k element from y_s_adv.
            v_s = -F.nll_loss(v_s, prediction_s.detach(), reduction='none')
            # picking element prediction_t k element from y_t_adv.
            v_t = -F.nll_loss(v_t, prediction_t.detach(), reduction='none')
        
        v_s_trans = self.tau_transform(v_s)
        v_t_trans = self.tau_transform(v_t)
        

        dst = self.gammaw * torch.mean(self.gf(v_s_trans)) - torch.mean(self.phistar_gf(v_t_trans))

        self.internal_stats['lhatsrc'] = torch.mean(v_s).item()
        self.internal_stats['lhattrg'] = torch.mean(v_t).item()
        self.internal_stats['acc'] = self.domain_discriminator_accuracy
        self.internal_stats['dst'] = dst.item()

        # we need to negate since the obj is being minimized, so min -dst =max dst.
        # the gradient reversar layer will take care of the rest
        return -self.multiplier * dst


# Example usage:
# if __name__ == "__main__":
#     # Create an affine transform with learnable parameters initialized to a=1.0, b=0.0
#     tau_affine = TauTransform(transform_type="affine", learnable=True, init_params={"a": 1.0, "b": 0.0})
#     # Create a power transform with fixed exponent c=0.5
#     tau_power = TauTransform(transform_type="power", learnable=False, init_params={"c": 0.5})
    
#     # Simulate input
#     x = torch.tensor([[0.1, 0.2, 0.3],
#                       [1.0, -1.0, 0.0]], dtype=torch.float32)
    
#     # Compute outputs
#     out_affine = tau_affine(x)
#     out_power = tau_power(x)
    
#     print("Affine transform output:\n", out_affine)
#     print("Power transform output:\n", out_power)
