from torch.cuda.tunable import enable
import torch
from typing import List, Callable, Union, Literal
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from torch import nn
import torch.nn.functional as F

from services.utils import update_ema
from services.tools import create_logger

logger = create_logger(__name__)


class TwinFlow(torch.nn.Module):
    def __init__(
        self,
        # --- Training Strategy & Consistency Control ---
        consistc_ratio: float = 1.0,
        ema_decay_rate: float = 0.99,
        # --- Enhanced Target Score Mechanism ---
        enhanced_ratio: float = 0.5,
        enhanced_range: List[float] = [0.00, 1.00],
        # --- Time Discretization & Distribution ---
        time_dist_ctrl: List[float] = [1.0, 1.0, 1.0],
        estimate_order: int = 2,
        loss_func_type: dict = {"type": "barron_reweighting"},
        dist_match_cof: float = 0.5,
        use_image_free: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.cor = consistc_ratio
        self.enr = enhanced_ratio
        self.emd = ema_decay_rate
        self.eng = enhanced_range
        self.tdc = time_dist_ctrl
        self.eso = estimate_order
        loss_func_args = loss_func_type.copy()
        self.lft = loss_func_args.pop("type")
        self.lfa = loss_func_args
        self.dmc = dist_match_cof
        self.uif = use_image_free

        assert self.eso >= 0, "Only support estimate_order >= 0"

        self.cmd = 0
        self.l2p = 0
        self.step = 0
        self.mod = None
        self.lsw = None
        self.nge = None

        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
            self.alpha_in, self.gamma_in = self.gamma_in, self.alpha_in
            self.alpha_to, self.gamma_to = self.gamma_to, self.alpha_to
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1

    def sample_beta(self, theta_1, theta_2, x):
        size = [x.size(0)] + [1] * (len(x.shape) - 1)
        beta_dist = torch.distributions.Beta(theta_1, theta_2)
        beta_samples = beta_dist.sample(size)
        return beta_samples.to(x)

    def l2_loss(self, pred, target):
        """
        Standard l2 Loss.
        """
        loss = (pred.float() - target.float()) ** 2
        return loss.flatten(1).mean(dim=1).to(pred.dtype)

    def barron_reweighting_loss(self, pred, target, alpha=1.0, c=1e-3):
        """
        Barron-Reweighted L2 Loss (Sample-level + Detach).
        """
        mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.ndim)))
        rmse_sq_norm = mse / (c**2 + 1e-8)  # (x/c)^2

        if abs(alpha - 2.0) < 1e-5:
            weight = torch.ones_like(mse)
        else:
            abs_a_m_2 = abs(alpha - 2.0)
            weight = (rmse_sq_norm / abs_a_m_2 + 1.0).pow((alpha - 2.0) / 2.0)
        loss = mse * weight.data

        return loss / c

    def loss_func(self, pred, target):
        loss_func = {
            "l2": self.l2_loss,
            "barron_reweighting": self.barron_reweighting_loss,
        }[self.lft]

        return loss_func(pred, target, **self.lfa)

    def prepare_inputs(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor],
    ):
        # 1. Init time and containers
        bsz, device = x.shape[0], x.device
        assert bsz >= 4 # we need minimal batch=4 to assign different target time, see L132
        t = self.sample_beta(self.tdc[0], self.tdc[1], x).clamp_(0, 1) * self.tdc[2]
        c = [torch.zeros_like(t)] if c is None else c
        e = [torch.zeros_like(t)] if e is None else e

        # Aux time variables
        tt = t - torch.rand_like(t) * self.cor * t
        t_min = t - torch.rand_like(t) * t * min(0.05, self.cor)

        probs = {"e2e": 1, "mul": 1, "any": 1, "adv": 1}
        # 3. Partition batch & Create masks
        keys, vals = list(probs.keys()), list(probs.values())
        lens = [int(round(bsz * v / (sum(vals) or 1))) for v in vals]
        lens[-1] = bsz - sum(lens[:-1])  # Adjust last to match bsz

        masks, cursor = {}, 0
        for k, l in zip(keys, lens):
            m = torch.zeros(bsz, dtype=torch.bool, device=device)
            if l > 0:
                m[cursor : cursor + l] = True
            masks[k] = m
            cursor += l

        # 4. Apply specific logic (using walrus operator for compactness)
        if (m := masks.get("e2e")) is not None and m.any():
            t[m], tt[m] = 1.0, 0.0
        if (m := masks.get("mul")) is not None and m.any():
            tt[m] = t_min[m]

        if self.uif:
            x_fake, _, _, _ = self.forward(
                model,
                torch.randn_like(x),
                torch.ones_like(t),
                torch.zeros_like(t),
                **dict(c=c),
            )
            x = x_fake.data

        # 5. Adversarial Generation Phase
        if (m := masks.get("adv")) is not None and m.any():
            tt[m] = -t[m]  # Flag logic
            # Generate fake samples (t=1 -> t=0) to replace inputs
            x_fake, _, _, _ = self.forward(
                model,
                torch.randn_like(x[m]),
                torch.ones_like(t[m]),
                torch.zeros_like(t[m]),
                **dict(c=[ic[m] for ic in c]),
            )
            x[m] = x_fake.data

        # 6. Construct Training Targets
        z = torch.randn_like(x)
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        target = z * self.alpha_to(t) + x * self.gamma_to(t)

        # Restore adv time sign for loss/matching logic
        if masks["adv"].any():
            t[masks["adv"]] *= -1
        return x_t, t, tt, c, e, target, masks

    @torch.no_grad()
    def get_refer_predc(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor],
    ):
        torch.cuda.set_rng_state(rng_state)
        refer_x, refer_z, refer_v, _ = self.forward(model, x_t, t, tt, **dict(c=e))
        torch.cuda.set_rng_state(rng_state)
        predc_x, predc_z, predc_v, _ = self.forward(model, x_t, t, tt, **dict(c=c))
        return refer_x, refer_z, refer_v, predc_x, predc_z, predc_v

    @torch.no_grad()
    def enhance_target(
        self,
        target: torch.Tensor,
        idx: torch.Tensor,
        ratio: float,
        pred_w_c: torch.Tensor,
        pred_wo_c: torch.Tensor,
    ):
        target[idx] = target[idx] + ratio * (pred_w_c[idx] - pred_wo_c[idx])
        # target[~idx] = (target[~idx] + pred_w_c[~idx]) * 0.50
        return target

    @torch.no_grad()
    def multi_fwd(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        N: int,
    ):
        pred = 0
        ts = [t * (1 - i / (N)) + tt * (i / (N)) for i in range(N + 1)]
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            torch.cuda.set_rng_state(rng_state)
            hx, hz, F_c, _ = self.forward(model, x_t, t_c, t_n, **dict(c=c))
            x_t = self.alpha_in(t_n) * hz + self.gamma_in(t_n) * hx
            pred = pred + F_c * (t_c - t_n)
        return hx, hz, pred

    @torch.no_grad()
    def get_rcgm_target(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        F_th_t: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        N: int,
    ):
        """
        References:
        - RCGM: https://github.com/LINs-lab/RCGM/blob/main/assets/paper.pdf
        """
        t_m = (t - 0.01).clamp_min(tt)
        x_t = x_t - target * 0.01
        _, _, Ft_tar = self.multi_fwd(rng_state, model, x_t, t_m, tt, c, N)
        mask = t < (tt + 0.01)
        cof_l = torch.where(mask, torch.ones_like(t), 100 * (t - tt))
        cof_r = torch.where(mask, 1 / (t - tt), torch.ones_like(t) * 100)
        Ft_tar = (F_th_t * cof_l - Ft_tar * cof_r) - target
        Ft_tar = F_th_t.data - (Ft_tar).clamp(min=-1.0, max=1.0)
        return Ft_tar

    @torch.no_grad()
    def update_ema(
        self,
        model: nn.Module,
    ):
        if self.emd > 0.0 and self.emd < 1.0:
            assert hasattr(model, "ema_transformer")
            self.mod = self.mod or model.ema_transformer
            update_ema(self.mod, model.transformer, decay=self.cmd)
            self.cmd += (1 - self.cmd) * (self.emd - self.cmd) * 0.5
        elif self.emd == 0.0:
            self.mod = model.transformer
        elif self.emd == 1.0:
            self.mod = model.ema_transformer

    @torch.no_grad()
    def dist_match(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: List[torch.Tensor],
    ):
        z = torch.randn_like(x)
        t = self.sample_beta(1.0, 1.0, x)
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        fake_s, _, fake_v, _ = self.forward(model, x_t, -t, -t, **dict(c=c))
        real_s, _, real_v, _ = self.forward(model, x_t, t, t, **dict(c=c))
        F_grad = fake_v - real_v
        x_grad = fake_s - real_s

        F_grad = F_grad.to(torch.float32).clamp(min=-1.0, max=1.0)
        x_grad = x_grad.to(torch.float32).clamp(min=-1.0, max=1.0)
        return x_grad, F_grad

    def training_step(
        self,
        model: Union[nn.Module, Callable],
        x: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor],
        step: int,
        v: torch.Tensor,
    ):
        with torch.no_grad():
            x_t, t, tt, c, e, target, sample_masks = self.prepare_inputs(model, x, c, e)

        loss, rng_state = 0, torch.cuda.get_rng_state()
        x_wc_t, z_wc_t, F_th_t, den_t = self.forward(model, x_t, t, tt, **dict(c=c))

        # Start update the state of ema model
        if (self.enr != 0.0) or (self.cor != 0.0) or (self.emd != 0.0):
            self.update_ema(model)

        # Start enhancing target
        if (self.eso >= 0) and (self.enr != 0.0):
            _, _, refer_v, _, _, predc_v = self.get_refer_predc(
                rng_state, self.mod, x_t, t, t, c, e
            )
            idx = (t.flatten() < self.eng[1]) & (t.flatten() > self.eng[0])
            idx = idx & (~sample_masks["adv"])
            target_ = target.clone()
            target = self.enhance_target(target, idx, self.enr, predc_v, refer_v)
            if self.emd == 1.0:
                target[idx] = (target - target_)[idx] + refer_v[idx]

        # Start optimizing RCGM-based or multi-step generation (if self.eso = 0)
        rcgm_idx = sample_masks["e2e"] | sample_masks["any"]
        if (self.eso >= 1) and (self.cor != 0.0) and (rcgm_idx.any()):
            model_4_rcgm = self.mod if (self.emd != 1.0) else model
            rcgm_target = self.get_rcgm_target(
                rng_state,
                model_4_rcgm,
                F_th_t,
                target,
                x_t,
                t,
                tt,
                c,
                self.eso,
            )
            rcgm_idx = ~(sample_masks["adv"])
            target[rcgm_idx] = rcgm_target[rcgm_idx]

        weighting = (torch.tan((1 - (t.abs() - tt.abs())) * np.pi / 2.5) + 1).flatten()
        weighting[sample_masks["adv"]] = 1
        loss = self.loss_func(F_th_t, target) * weighting
        # Only calculate the multi-step generation loss (if self.eso = 0)
        mul_adv_idx = sample_masks["mul"] | sample_masks["adv"]
        if self.eso >= 1:
            loss = loss.mean()
        elif mul_adv_idx.any():
            loss = loss[mul_adv_idx].mean()
        else:
            loss = 0

        # Start optimizing one or few-step generation
        opt_idx = sample_masks["e2e"]
        if (self.cor == 1.0) and (opt_idx.any()):
            c_e2e = [ic[opt_idx] for ic in c]
            x_e2e = x_wc_t[opt_idx]
            _, F_grad = self.dist_match(model, x_e2e, c_e2e)
            F_fake = F_th_t[opt_idx]
            e2e_loss = self.loss_func(F_fake, (F_fake - F_grad).data)
            loss = loss + e2e_loss.mean() * self.dmc
        return loss

    def forward(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
        **model_kwargs,
    ):
        tt = tt.flatten()  # if tt is not None else t.clone().flatten()
        # dent = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        dent = -1
        q = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        q = q if self.integ_st == 1 else 1 - q
        F_t = (-1) ** (1 - self.integ_st) * model(x_t, t=q, tt=tt, **model_kwargs)
        t = torch.abs(t)
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def kumaraswamy_transform(self, t, a, b, c):
        return (1 - (1 - t**a) ** b) ** c

    @torch.no_grad()
    def sampling_loop(
        self,
        inital_noise_z: torch.FloatTensor,
        sampling_model: Union[nn.Module, Callable],
        sampling_steps: int = 20,
        stochast_ratio: float = 0.0,
        extrapol_ratio: float = 0.0,
        sampling_order: int = 1,
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
        rfba_gap_steps: list = [0.0, 0.0],
        sampling_style: Literal["few", "mul", "any"] = "few",
        **model_kwargs,
    ):
        """
        References:
        - UCGM (Sampler): https://arxiv.org/abs/2505.07447 (Unified Continuous Generative Models)
        """
        input_dtype = inital_noise_z.dtype
        assert sampling_order in [1, 2]
        num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps

        # Time step discretization.
        num_steps = num_steps + 1 if (rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
        t_steps = torch.linspace(
            rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64
        ).to(inital_noise_z)
        t_steps = t_steps[:-1] if (rfba_gap_steps[1] - 0.0) == 0.0 else t_steps
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)
        t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])

        # Prepare the buffer for the first order prediction
        x_hats, z_hats, buffer_freq = [], [], 1

        # Main sampling loop
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z.cpu()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # First order prediction
            if sampling_style == "few":
                t_tgt = torch.zeros_like(t_cur)
            elif sampling_style == "mul":
                t_tgt = t_cur
            elif sampling_style == "any":
                t_tgt = t_next
            else:
                raise ValueError(f"Unknown sampling style: {sampling_style}")

            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(input_dtype),
                t_cur.to(input_dtype),
                t_tgt.to(input_dtype),
                **model_kwargs,
            )
            samples.append(x_hat.cpu())
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

            # Apply extrapolation for prediction (extrapolating z is not nessary?)
            if buffer_freq > 0 and extrapol_ratio > 0:
                z_hats.append(z_hat)
                x_hats.append(x_hat)
                if i > buffer_freq:
                    z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                    x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                    z_hats.pop(0), x_hats.pop(0)

            if stochast_ratio == "SDE":
                stochast_ratio = (
                    torch.sqrt((t_next - t_cur).abs())
                    * torch.sqrt(2 * self.alpha_in(t_cur))
                    / self.alpha_in(t_next)
                )
                stochast_ratio = torch.clamp(stochast_ratio ** (1 / 0.50), min=0, max=1)
                noi = torch.randn(x_cur.size()).to(x_cur)
            else:
                noi = torch.randn(x_cur.size()).to(x_cur) if stochast_ratio > 0 else 0.0
            x_next = self.gamma_in(t_next) * x_hat + self.alpha_in(t_next) * (
                z_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
            )

            # Apply second order correction, Heun-like
            if sampling_order == 2 and i < num_steps - 1:
                x_pri, z_pri, _, _ = self.forward(
                    sampling_model,
                    x_next.to(input_dtype),
                    t_next.to(input_dtype),
                    **model_kwargs,
                )
                x_pri, z_pri = x_pri.to(torch.float64), z_pri.to(torch.float64)

                x_next = x_cur * self.gamma_in(t_next) / self.gamma_in(t_cur) + (
                    self.alpha_in(t_next)
                    - self.gamma_in(t_next)
                    * self.alpha_in(t_cur)
                    / self.gamma_in(t_cur)
                ) * (0.5 * z_hat + 0.5 * z_pri)

            x_cur = x_next

        return torch.stack(samples, dim=0).to(input_dtype)
