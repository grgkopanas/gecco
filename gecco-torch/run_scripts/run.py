import sys
import os
#from shapenet_airplane_unconditional import data, model
from configurations.common import prepare_trainer
from configurations.gaussian_splatting import prepare_data, prepare_model
from configurations.arguments import get_args
from einops import rearrange
import torch

def dry_run_for_stats(model, data):
    batches = []
    data.setup()
    dataloader = data.train_dataloader()


    for i, batch in enumerate(dataloader):
        if i == 6: # break early to save time
            break
        batches.append(batch)

    batches = dataloader.collate_fn(batches) # [5, batch_size, ...]
    batches = batches.apply_to_tensors(lambda t: t.reshape(-1, *t.shape[2:])) # [5 * batch_size, ...]

    max_dx = (batches.data[:, :, 0].max(dim=1)[0] - batches.data[:, :, 0].min(dim=1)[0]).max()
    max_dy = (batches.data[:, :, 1].max(dim=1)[0] - batches.data[:, :, 1].min(dim=1)[0]).max()
    max_dz = (batches.data[:, :, 2].max(dim=1)[0] - batches.data[:, :, 2].min(dim=1)[0]).max()

    print(f"[UNPREPARED] Standard Deviation: {batches.data.std(dim=(0,1))=}")
    print(f"[UNPREPARED] Mean: {batches.data.mean(dim=(0,1))=}")

    diff_adjusted = model.reparam.data_to_diffusion(batches.data, batches.ctx)
    
    print(f"[PREPARED] Standard Deviation: {diff_adjusted.std(dim=(0,1))=}")
    print(f"[PREPARED] Mean: {diff_adjusted.mean(dim=(0,1))=}")



    diff_flat = rearrange(diff_adjusted[:6], 'b n d -> (b n) d')
    print(diff_adjusted.shape)
    distm = torch.cdist(diff_flat, diff_flat)
    print(f"[PREPARED] {distm.max()=}")

    test_data = diff_adjusted.data[1:2].repeat(100,1,1)
    sigma = model.loss.schedule.test(test_data)
    
    n = torch.randn_like(test_data) * sigma

def train(args):
    model = prepare_model(args)
    data = prepare_data(args)
    trainer = prepare_trainer(args)

    #model = torch.compile(model)
    if True:
        trainer.fit(model, data)
    else:
        dry_run_for_stats(model, data)

if __name__ == "__main__":
    args = get_args()
    train(args)
