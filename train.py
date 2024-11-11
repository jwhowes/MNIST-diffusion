import torch
import os
import torch.nn.functional as F

from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.model import ClassConditionalVitDiffuser, ClassConditionalConvNeXtUNetDiffuser
from src.data import MNISTDataset


def train(model, dataloader):
    num_epochs = 5
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=500, num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    model, dataloader, opt, lr_scheduler = accelerator.prepare(
        model, dataloader, opt, lr_scheduler
    )

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}")
        for i, (image, label) in enumerate(dataloader):
            opt.zero_grad()

            loss = model(image, label)

            loss.backward()
            opt.step()
            lr_scheduler.step()

            if i % 10 == 0:
                print(f"\t{i} / {len(dataloader)} iters. Loss: {loss.item():.6f}")

        torch.save(
            accelerator.get_state_dict(model, unwrap=True),
            os.path.join("checkpoints", f"checkpoint_{(epoch + 1):03}.pt")
        )


if __name__ == "__main__":
    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    model = ClassConditionalConvNeXtUNetDiffuser(
        d_init=128,
        d_t=256,
        n_scales=2,
        n_classes=10,
        n_channels=1,
        n_blocks_per_scale=2,
        t_min=0.01,
        t_max=3.5,
        p_uncond=0.1
    )

    dataset = MNISTDataset(split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    train(model, dataloader)
