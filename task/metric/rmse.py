from torchmetrics import Metric
import torch


class RootMeanSquaredError(Metric):

    def __init__(self, lc_min: int, lc_max: int, target_length: int, **kwargs):
        super().__init__()

        # State variables for the metric
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("batch_idx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.target_length = target_length

        self.lc_min = lc_min
        self.lc_max = lc_max
        print(
            f"Using Masked RootMeanSquaredError metric Loss with Landcover boundaries ({self.lc_min, self.lc_max})."
        )

    def update(self, preds, targs):
        targets = targs["dynamic"][0][:, -self.target_length:, 0, ].unsqueeze(2)
        preds = preds[:, -self.target_length:, 0, ].unsqueeze(2)
        s2_mask = (targs["dynamic_mask"][0][:, -self.target_length:] < 1.0).bool().type_as(preds)

        lc = targs["landcover"]
        lc_mask = (
            ((lc <= self.lc_min).bool() | (lc >= self.lc_max).bool())
            .type_as(s2_mask)
            .unsqueeze(1)
            .repeat(1, preds.shape[1], 1, 1, 1)
        )  # b t c h w

        mask = s2_mask * lc_mask

        sum_squared_error = torch.pow((preds - targets) * mask, 2).sum()
        n_obs = (mask == 1).sum()

        # Update the states variables

        self.sum_squared_error += sum_squared_error / (n_obs + 1e-8)
        self.batch_idx += 1

    def compute(self):
        """
        Compute the final Root Mean Squared Error (RMSE) over the state of the metric.

        Returns
        -------
        dict
            Dictionary containing the computed RMSE for vegetation pixels.
        """
        return {"RMSE_Veg": torch.sqrt(self.sum_squared_error / self.batch_idx)}

