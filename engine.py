import torch
# import numpy as np
# import math
import torch.nn.functional as F
# from einops import rearrange
# from torchvision.utils import make_grid

from util.evaluator import img_metrics, avg_img_metrics
from util.cknna import AlignmentMetrics

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class CKNNAAvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def update(self, score):
        # score 可以是 float / tensor / dict
        if isinstance(score, dict):
            score = next(iter(score.values()))
        if hasattr(score, "item"):
            score = score.item()
        self.sum += score
        self.n += 1

    def value(self):
        avg_score = self.sum / max(self.n, 1)
        return {'CKNNA': avg_score}


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

        self.avg_cknna = CKNNAAvgMeter()
        self.avg_miou = mIoUAvgMeter(num_classes=6)

        assert self.method in ["heun", "euler"]
        assert self.prediction in ["e", "v", "x"]
        assert self.loss in ["e", "v", "x"]

    def scale_01(self, batch_image):
        return (batch_image + 1.0) / 2.0

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def get_proj_loss(self, zs, zs_tilde):
        proj_loss = 0.
        bsz = zs[0].shape[0]
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                z_j = torch.nn.functional.normalize(z_j, dim=-1)
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
        proj_loss /= (len(zs) * bsz)
        return proj_loss

    def __call__(self, batch):
        x, cond = batch['clear'].clone(), batch['cloudy'].clone()
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        # v = (x - z) / (1 - t).clamp_min(self.t_eps)
        # my calculate v:
        v = x - e
        # e = (z - x * t) / (1 - t).clamp_min(self.t_eps)

        # x-pred
        if self.prediction == "x":
            x_pred, fs = self.model(z, t.flatten(), cond, return_f=True)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
            e_pred = (z - x_pred * t) / (1 - t).clamp_min(self.t_eps)
        # v-pred
        elif self.prediction == "v":
            v_pred, fs = self.model(z, t.flatten(), cond, return_f=True)
            x_pred = (1 - t) * v_pred + z
            e_pred = z - t * v_pred
        # e-pred
        elif self.prediction == "e":
            e_pred, fs = self.model(z, t.flatten(), cond, return_f=True)
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

        zs = batch['zs']
        proj_loss = self.get_proj_loss(zs, fs)

        return denoising_loss, proj_loss

    @torch.no_grad()
    def test_step(self, batch):
        x, cond = batch['clear'].clone(), batch['cloudy'].clone()
        f_target = batch['zs']
        device = cond.device
        bsz = cond.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        # samples = self.sample(cond, z, bsz=bsz, device=device)
        samples, fs = self.sample(cond, z, bsz=bsz, device=device, return_f=True)
        for i in range(samples.shape[0]):
            _target = x[i, ...]
            _samples = samples[i, ...]
            _target = self.scale_01(_target)
            _samples = self.scale_01(_samples)
            metrics = img_metrics(target=_target.unsqueeze(0), pred=_samples.unsqueeze(0))
            self.avg_metrics.add(metrics)

            # feats_A = F.normalize(fs[i].reshape(-1, fs[i].shape[-1]), dim=-1)
            feats_A = F.normalize(fs[i], dim=-1)
            feats_B = F.normalize(f_target[i], dim=-1)
            cknna_score = AlignmentMetrics.measure('cknna', feats_A, feats_B, topk=10)
            self.avg_cknna.update(cknna_score)

        return self.avg_metrics

    @torch.no_grad()
    def sample(self, cond, z, bsz, device, return_f=False):
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, cond)
        # last step euler
        if return_f:
            z, f = self._euler_step(z, timesteps[-2], timesteps[-1], cond, return_f)
            return z, f
        else:
            z = self._euler_step(z, timesteps[-2], timesteps[-1], cond)
            return z

    @torch.no_grad()
    def _forward_sample(self, z, t, cond, return_f=False):
        if return_f:
            if self.prediction == "x":
                x_cond, f = self.model(z, t.flatten(), cond, return_f)
                v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
            elif self.prediction == "v":
                v_cond, f = self.model(z, t.flatten(), cond, return_f)
            elif self.prediction == "e":
                e_cond, f= self.model(z, t.flatten(), cond, return_f)
                v_cond = (z - e_cond) / t.clamp_min(self.t_eps)
            return v_cond, f
        else:
            if self.prediction == "x":
                x_cond = self.model(z, t.flatten(), cond)
                v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
            elif self.prediction == "v":
                v_cond = self.model(z, t.flatten(), cond)
            elif self.prediction == "e":
                e_cond = self.model(z, t.flatten(), cond)
                v_cond = (z - e_cond) / t.clamp_min(self.t_eps)
            return v_cond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond, return_f=False):
        if return_f:
            v_pred, f = self._forward_sample(z, t, cond, return_f)
            z_next = z + (t_next - t) * v_pred
            return z_next, f
        else:
            v_pred = self._forward_sample(z, t, cond)
            z_next = z + (t_next - t) * v_pred
            return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond):
        v_pred_t = self._forward_sample(z, t, cond)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, cond)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

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
            samples = self.sample(cond, z, bsz=bsz, device=device)
            results["samples"] = self.scale_01(samples)

        return results
