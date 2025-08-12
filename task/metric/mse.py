import numpy as np
from torchmetrics import Metric
import torch
from skimage.metrics import structural_similarity as compare_ssim
import lpips
loss_fn_alex = lpips.LPIPS(net='alex')
class MSE(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(self, target_length: int,):

        super().__init__()
        # State variables for the metric
        self.add_state("sum_mse_error_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        # self.add_state("sum_lpips_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_psnr_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_ssim_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_mae_t", default=torch.zeros(target_length), dist_reduce_fx="sum")
        self.add_state("sum_idx", default=torch.tensor(0), dist_reduce_fx="sum")

        self.target_length = target_length


    def update(self, preds, targs, test=False):
        targs = targs[:, -self.target_length:]
        preds = preds[:, -self.target_length:]
        preds = torch.clip(preds, min=0, max=1)
        # The frame-by-frame error is calculated
        self.sum_idx += 1

        pixel_error = preds - targs
        mse_mean_t = (pixel_error**2).mean(dim=0).sum([1,2,3])
        self.sum_mse_error_t += mse_mean_t


        if test:
            mae_mean_t = torch.abs(pixel_error).mean(dim=0).sum([1, 2, 3])
            self.sum_mae_t += mae_mean_t
            b, t, c, h, w = preds.shape
            multichannel = True if c > 1 else False
            preds_t = preds.reshape(-1, c, h, w)
            targs_t = targs.reshape(-1, c, h, w)
            # ssim_t and psnr
            if multichannel:
                real_frm = np.transpose(targs_t.cpu().numpy(), (0, 2, 3, 1))
                pred_frm = np.transpose(preds_t.cpu().numpy(), (0, 2, 3, 1))
                real_lp = targs_t.cpu()
                pred_lp = preds_t.cpu()

            else:
                real_frm = targs_t.squeeze(1).cpu().numpy()
                pred_frm = preds_t.squeeze(1).cpu().numpy()
                real_lp = targs_t.repeat(1, 3, 1, 1).cpu()
                pred_lp = preds_t.repeat(1, 3, 1, 1).cpu()

            # lp_loss = loss_fn_alex(real_lp, pred_lp).view(b, t).mean(dim=0).to(targs_t.device)
            # self.sum_lpips_t += lp_loss

            self.sum_psnr_t += (torch.tensor(self.batch_psnr(np.uint8(pred_frm*255), np.uint8(real_frm*255)), device=targs_t.device).view(b, t)
                            .mean(dim=0))

            ssim_t = torch.zeros(preds_t.shape[0], device=preds_t.device)
            for i in range(preds_t.shape[0]):
                score, _ = compare_ssim(pred_frm[i], real_frm[i], full=True,  multichannel=multichannel)
                ssim_t[i] += torch.tensor(score, device=preds_t.device)

            self.sum_ssim_t += ssim_t.view(b, t).mean(dim=0)

    def batch_psnr(self, gen_frames, gt_frames):
        if gen_frames.ndim == 3:
            axis = (1, 2)
        elif gen_frames.ndim == 4:
            axis = (1, 2, 3)
        x = np.int32(gen_frames)
        y = np.int32(gt_frames)
        num_pixels = float(np.size(gen_frames[0]))
        mse = np.sum((x - y) ** 2, axis=axis, dtype=np.float32) / num_pixels
        psnr = 20 * np.log10(255) - 10 * np.log10(mse)
        return psnr


    def compute(self, validation=True,):
        if validation:
            return {"MSE": self.sum_mse_error_t.mean() / self.sum_idx}
        else:
            score = {"MSE": self.sum_mse_error_t.mean() / self.sum_idx,
                     "MAE": self.sum_mae_t.mean() / self.sum_idx,
                     "ssim": self.sum_ssim_t.mean() / self.sum_idx,
                     "psnr": self.sum_psnr_t.mean() / self.sum_idx,
                     # "lpips": self.sum_lpips_t.mean() / self.sum_idx,
                     }

            score_t = {"MSE_t": self.sum_mse_error_t / self.sum_idx,
                       "MAE_t": self.sum_mae_t / self.sum_idx,
                       "ssim_t": self.sum_ssim_t / self.sum_idx,
                       "psnr_t": self.sum_psnr_t / self.sum_idx,
                       # "lpips_t": self.sum_lpips_t / self.sum_idx,
                    }
            return score, score_t
