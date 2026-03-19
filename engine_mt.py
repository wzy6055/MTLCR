import torch
import torch.nn.functional as F
import torch.nn as nn

from util.evaluator import img_metrics, avg_img_metrics

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class mIoUAvgMeter:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confmat = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.float64
        )

    @torch.no_grad()
    def update(self, pred, target):
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)

        pred = pred.view(-1)
        target = target.view(-1)

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        idx = target * self.num_classes + pred
        cm = torch.bincount(
            idx,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        if self.confmat.device != cm.device:
            self.confmat = self.confmat.to(cm.device)
        self.confmat += cm

    def value(self):
        tp = torch.diag(self.confmat)
        fp = self.confmat.sum(dim=0) - tp
        fn = self.confmat.sum(dim=1) - tp

        iou = tp / (tp + fp + fn + 1e-6)
        miou = iou.mean().item()

        return {'mIoU': miou}

class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class JiTEngine:
    def __init__(
            self,
            model,
            config,
    ):
        self.model = model
        self.steps = config.sampling.num_sampling_steps
        self.method = config.sampling.sampling_method
        self.prediction = config.sampling.prediction
        self.loss = config.sampling.loss
        self.img_size = config.dataset.resolution
        self.avg_metrics = avg_img_metrics()
        self.P_mean = config.sampling.P_mean
        self.P_std = config.sampling.P_std
        self.noise_scale = config.sampling.noise_scale
        self.t_eps = config.sampling.t_eps

        self.avg_miou = mIoUAvgMeter(num_classes=6)

        assert self.method in ["heun", "euler"]
        assert self.prediction in ["e", "v", "x"]
        assert self.loss in ["e", "v", "x"]

    def scale_01(self, batch_image):
        return (batch_image + 1.0) / 2.0

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def __call__(self, batch):
        x, cond = batch['clear'].clone(), batch['cloudy'].clone()
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        # v = (x - z) / (1 - t).clamp_min(self.t_eps)
        # my calculate v:
        v = x - e
        # e = (z - x * t) / (1 - t).clamp_min(self.t_eps)

        output = self.model(z, t.flatten(), cond)

        # x-pred
        if self.prediction == "x":
            x_pred = output['x']
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
            e_pred = (z - x_pred * t) / (1 - t).clamp_min(self.t_eps)
        # v-pred
        elif self.prediction == "v":
            v_pred = output['x']
            x_pred = (1 - t) * v_pred + z
            e_pred = z - t * v_pred
        # e-pred
        elif self.prediction == "e":
            e_pred = output['x']
            x_pred = (z - (1-t) * e_pred) / t.clamp_min(self.t_eps)
            v_pred = (z - e_pred) / t.clamp_min(self.t_eps)

        # x-loss
        if self.loss == "x":
            loss = (x - x_pred) ** 2
        # v-loss
        elif self.loss == "v":
            loss = (v - v_pred) ** 2
        # e-loss
        elif self.loss == "e":
            loss = (e - e_pred) ** 2

        # loss = loss.mean(dim=(1, 2, 3)).mean()
        denoising_loss = mean_flat(loss)

        return denoising_loss

    @torch.no_grad()
    def test_step(self, batch):
        x, cond = batch['clear'].clone(), batch['cloudy'].clone()
        device = cond.device
        bsz = cond.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        sample_state = {
            'cond': cond,
            'z_next': z,
            'bsz': bsz,
            'device': device,
        }
        sample_state = self.sample(sample_state)
        samples = sample_state['z_next']
        for i in range(samples.shape[0]):
            _target = x[i, ...]
            _samples = samples[i, ...]
            _target = self.scale_01(_target)
            _samples = self.scale_01(_samples)
            metrics = img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            self.avg_metrics.add(metrics)

        return self.avg_metrics

    @torch.no_grad()
    def sample(self, state):
        cond = state['cond']
        z = state['z_next']
        bsz = state['bsz']
        device = state['device']
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        output = dict(state)
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError
        for i in range(self.steps - 1):
            output['t'] = timesteps[i]
            output['t_next'] = timesteps[i + 1]
            output = stepper(output)
        # last step euler
        output['t'] = timesteps[-2]
        output['t_next'] = timesteps[-1]
        output = self._euler_step(output)
        return output

    @torch.no_grad()
    def _forward_sample(self, state):
        z = state['z_next']
        t = state['t']
        cond = state['cond']
        if self.prediction == "x":
            output = self.model(z, t.flatten(), cond)
            x_cond = output['x']
            v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
            output['v_cond'] = v_cond
        elif self.prediction == "v":
            output = self.model(z, t.flatten(), cond)
            output['v_cond'] = output['x']
        elif self.prediction == "e":
            output = self.model(z, t.flatten(), cond)
            e_cond = output['x']
            output['v_cond'] = (z - e_cond) / t.clamp_min(self.t_eps)
        merged_state = dict(state)
        merged_state.update(output)
        return merged_state

    @torch.no_grad()
    def _euler_step(self, state):
        output = self._forward_sample(state)
        z = output['z_next']
        t = output['t']
        t_next = output['t_next']
        v_pred = output['v_cond']
        output['z_next'] = z + (t_next - t) * v_pred
        return output

    @torch.no_grad()
    def _heun_step(self, state):
        output_t = self._forward_sample(state)
        v_pred_t = output_t['v_cond']
        z = output_t['z_next']
        t = output_t['t']
        t_next = output_t['t_next']

        z_next_euler = z + (t_next - t) * v_pred_t
        next_state = dict(output_t)
        next_state['z_next'] = z_next_euler
        next_state['t'] = t_next
        output_t_next = self._forward_sample(next_state)
        v_pred_t_next = output_t_next['v_cond']

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        output = dict(output_t_next)
        output['v_cond'] = v_pred
        output['z_next'] = z + (t_next - t) * v_pred
        return output

    @torch.no_grad()
    def log_images(self, batch, sample=True):
        results = dict()
        results["input"] = self.scale_01(batch['clear'].clone().detach())
        results["cloudy"] = self.scale_01(batch['cloudy'].clone().detach())

        x, cond = batch['clear'].clone(), batch['cloudy'].clone()
        device = cond.device
        bsz = cond.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)

        if sample:
            sample_state = {
                'cond': cond,
                'z_next': z,
                'bsz': bsz,
                'device': device,
            }
            sample_state = self.sample(sample_state)
            samples = sample_state['z_next']
            results["samples"] = self.scale_01(samples)

        return results
