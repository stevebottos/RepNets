# pylint: disable=not-callable
import time

import torch
from torchmetrics.classification import Accuracy
from tqdm import tqdm

from repnets.datasets import cifar10, cifar100
from repnets.models import simple_net
from repnets.models.repvgg import get_RepVGG_func_by_name
from repnets.models.repshufflenetv2 import shufflenet_v2_x0_5

N_EPOCHS = 100

# Get the CIFAR10 dataset, because it's readily available and simple
train_dataloader, val_dataloader, labelmap = cifar100.get(
    train_batchsize=16,
    val_batchsize=16,
    num_workers=4,
)

# model = get_RepVGG_func_by_name("RepVGG-A0")(num_classes=len(labelmap))
# model = simple_net.SimpleNet(num_classes=len(labelmap))
model = shufflenet_v2_x0_5(num_classes=len(labelmap))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
metric = Accuracy(task="multiclass", num_classes=len(labelmap))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = criterion.to(device)
metric = metric.to(device)

for epoch in range(N_EPOCHS):
    model.train()
    for image_batch, label_batch in tqdm(train_dataloader, ncols=60):
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        predictions = model(image_batch)
        loss = criterion(predictions, label_batch)
        metric.update(predictions, label_batch)
        loss.backward()
        optimizer.step()

    train_accuracy = round(metric.compute().item(), 3)
    metric.reset()

    model.eval()
    times = []
    with torch.no_grad():
        for image_batch, label_batch in tqdm(val_dataloader, ncols=60):
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            # Time the un-reparameterized model
            _t = time.time()
            predictions = model(image_batch)
            times.append(time.time() - _t)

            loss = criterion(predictions, label_batch)
            metric.update(predictions, label_batch)

    val_accuracy = round(metric.compute().item(), 3)
    unrep_time = round(sum(times) / len(times), 6)
    metric.reset()

    print(
        f"Epoch {epoch}: Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}\n"
    )

# Reparameterize the model and do another run through.
# Model is in eval mode already but for clarity we'll make sure
model.eval().reparameterize()
times = []
with torch.no_grad():
    for image_batch, label_batch in tqdm(val_dataloader, ncols=60):
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        # Time the reparameterized model
        _t = time.time()
        predictions = model(image_batch)
        times.append(time.time() - _t)

        loss = criterion(predictions, label_batch)
        metric.update(predictions, label_batch)

    reparameterized_accuracy = round(metric.compute().item(), 3)

rep_time = round(sum(times) / len(times), 6)
print(f"Final Original Accuracy: {val_accuracy}, Mean Inference Time: {unrep_time}")
print(
    f"Final Reparameterized Accuracy: {reparameterized_accuracy}, Mean Inference Time: {rep_time}"
)
