import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune


# =========================
#  Fisher Information Matrix
# =========================

class FisherInformationMatrix:
    def __init__(self, model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: torch.utils.data.DataLoader,
                 complete_fim: bool = True,
                 layers: list = None,
                 mask: dict = None,
                 sampling_type: str = 'complete',
                 sampling_frequency: tuple = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.layers = layers
        self.complete_fim = complete_fim
        self.mask = mask
        self.sampling_type = sampling_type
        self.sampling_frequency = sampling_frequency

        if self.layers is None and not self.complete_fim:
            raise ValueError("Either 'layers' must be specified or 'complete_fim' must be True.")

        self.sampling_masks = self._make_sampling_masks(model)
        self.concat_mask = self._make_concat_mask(model)

        if complete_fim:
            n_params = (self.concat_mask.sum().item()
                        if self.concat_mask is not None
                        else sum(p.numel() for p in model.parameters() if p.requires_grad))
            self.fim = {'complete': torch.zeros((n_params, n_params), device=self.device)}
        else:
            self.fim = {
                name: torch.zeros((param.numel(), param.numel()), device=self.device)
                for name, param in model.named_parameters()
                if param.requires_grad and name in self.layers
            }

        self.compute_fim(model.to(self.device), dataloader, optimizer)
        self.compute_logdet_metrics()

    def _make_sampling_masks(self, model):
        if self.sampling_type == 'complete':
            return None

        elif self.sampling_type == 'x_in_x':
            sampling_masks = {}
            for name, param in model.named_parameters():
                if self.layers is None or name in self.layers:
                    mask = [True if i % self.sampling_frequency[0] == 0 else False
                            for i in range(0, param.numel())]
                    sampling_masks[name] = torch.tensor(mask, dtype=torch.bool, device=self.device)
            return sampling_masks

        elif self.sampling_type == 'x_skip_y':
            sampling_masks = {}
            cycle_length = sum(self.sampling_frequency)
            for name, param in model.named_parameters():
                if self.layers is None or name in self.layers:
                    mask = (torch.arange(param.numel(), device=self.device) % cycle_length) < self.sampling_frequency[0]
                    sampling_masks[name] = mask.to(torch.bool)
            return sampling_masks
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_type}")

    def _make_concat_mask(self, model):
        if self.complete_fim is False:
            return None

        if self.mask is None:
            return torch.ones(
                sum(p.numel() for p in model.parameters() if p.requires_grad),
                device=self.device,
                dtype=torch.bool
            )

        concat_mask = torch.tensor([], device=self.device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.mask.keys():
                concat_mask = torch.cat((concat_mask,
                                         self.mask[name].view(-1).to(self.device)))
            else:
                concat_mask = torch.cat((concat_mask,
                                         torch.ones(param.numel(), device=self.device)))
        return concat_mask.to(torch.bool)

    def _compute_fim_complete(self, model, dataloader, optimizer) -> dict:
        model.eval()
        eps = 1e-8

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in self.mask.keys():
                            param.grad *= self.mask[name]

            grad = torch.cat([p.grad.view(-1) for p in model.parameters()
                              if p.requires_grad]).view(-1)
            self.fim['complete'] += torch.outer(grad[self.concat_mask],
                                                grad[self.concat_mask])

        self.fim['complete'] = self.fim['complete'] / len(dataloader.dataset)
        self.fim['complete'] += eps * torch.eye(self.fim['complete'].shape[0],
                                                dtype=self.fim['complete'].dtype,
                                                device=self.device)

    def _compute_fim_layerwise(self, model, dataloader, optimizer) -> dict:
        model.eval()
        eps = 1e-8

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in self.mask.keys():
                            param.grad *= self.mask[name]

            for name, param in model.named_parameters():
                if param.requires_grad and name in self.layers:
                    grad = param.grad.view(-1)
                    self.fim[name] += torch.outer(grad, grad)

        for name in self.fim:
            self.fim[name] = self.fim[name] / len(dataloader.dataset)
            self.fim[name] += eps * torch.eye(self.fim[name].shape[0],
                                              dtype=self.fim[name].dtype,
                                              device=self.device)

    def compute_fim(self, model, dataloader, optimizer) -> dict:
        if self.complete_fim:
            self._compute_fim_complete(model, dataloader, optimizer)
        else:
            self._compute_fim_layerwise(model, dataloader, optimizer)

    def compute_logdet_metrics(self):
        if self.complete_fim:
            fim = self.fim['complete']
            self.logdet = torch.slogdet(fim)[-1].item()
            self.diaglogdet = torch.sum(torch.log(torch.diag(fim))).item()
            self.logdet_ratio = self.diaglogdet - self.logdet
        else:
            logdets = {}
            diaglogdets = {}
            logdet_ratios = {}
            for name, fim in self.fim.items():
                logdet = torch.slogdet(fim)[-1].item()
                diaglogdet = torch.sum(torch.log(torch.diag(fim))).item()
                logdet_ratio = diaglogdet - logdet

                logdets[name] = logdet
                diaglogdets[name] = diaglogdet
                logdet_ratios[name] = logdet_ratio

            self.logdet = logdets
            self.diaglogdet = diaglogdets
            self.logdet_ratio = logdet_ratios


# =========================
#  Model & Data
# =========================

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.net = nn.Sequential( # input [batch_size, 1, 28, 28]
            nn.Conv2d(1, 4, 3),   # [batch_size, 4, 26, 26]
            nn.ReLU(),            #
            nn.MaxPool2d(2),      # [batch_size, 4, 13, 13]
            nn.Conv2d(4, 16, 4),   # [batch_size, 16, 10, 10]
            nn.ReLU(),            #  
            nn.MaxPool2d(2),      # [batch_size, 16, 5, 5]
            nn.Flatten(),         # [batch_size, 400]
            nn.Linear(400,10)     # output [batch_size, 10]
        )
    def forward(self, X):
        return self.net(X)


def get_mnist_loaders(batch_size=128, fim_subset_size=6000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fim_indices = list(range(min(fim_subset_size, len(train_dataset))))
    fim_subset = Subset(train_dataset, fim_indices)
    fim_loader = DataLoader(fim_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, fim_loader


# =========================
#  Training / Evaluation
# =========================

def train_model(model, train_loader, epochs=3, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    return model, optimizer, criterion


def evaluate(model, data_loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# =========================
#  Pruning Baselines
# =========================

def estimate_hessian_diag(model, dataloader, criterion, num_batches=None, device="cuda"):
    model.eval()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    h_diag = [torch.zeros_like(p, device=device) for p in params]

    for i, (x, y) in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        for H, p in zip(h_diag, params):
            if p.grad is None:
                continue
            H += p.grad.detach() ** 2

    denom = (num_batches if num_batches is not None else (i + 1))
    h_diag = [H / max(denom, 1) for H in h_diag]
    return h_diag


def obd_prune(model, dataloader, criterion, sparsity=0.9, num_batches=10, device="cuda"):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    h_diag = estimate_hessian_diag(model, dataloader, criterion, num_batches, device)

    scores = []
    for p, H in zip(params, h_diag):
        scores.append(0.5 * (p.data ** 2) * H)
    all_scores = torch.cat([s.view(-1) for s in scores])

    k = int(sparsity * all_scores.numel())
    if k <= 0:
        return model

    threshold, _ = torch.kthvalue(all_scores, max(k, 1))

    with torch.no_grad():
        for p, s in zip(params, scores):
            mask = (s > threshold).to(p.device)
            p.mul_(mask)

    return model


def structured_filter_prune(model, amount_per_layer=0.5):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module,
                                name="weight",
                                amount=amount_per_layer,
                                n=1,
                                dim=0)
            prune.remove(module, "weight")
    return model


def snip_prune(model, dataloader, criterion, sparsity=0.9, num_batches=1, device="cuda"):
    model.to(device)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    for i, (x, y) in enumerate(dataloader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

    scores = [torch.abs(p.grad * p) for p in params]
    all_scores = torch.cat([s.view(-1) for s in scores])

    k = int(sparsity * all_scores.numel())
    if k <= 0:
        return model

    threshold, _ = torch.kthvalue(all_scores, max(k, 1))

    with torch.no_grad():
        for p, s in zip(params, scores):
            mask = (s > threshold).to(p.device)
            p.mul_(mask)

    return model


# =========================
#  Helper: save FIM
# =========================

def save_fim(fim_obj: FisherInformationMatrix, baseline_name: str, out_dir: str = "fims"):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{baseline_name}_fim.pt")

    to_save = {
        "fim": fim_obj.fim,
        "logdet": fim_obj.logdet,
        "diaglogdet": fim_obj.diaglogdet,
        "logdet_ratio": fim_obj.logdet_ratio,
    }
    torch.save(to_save, save_path)
    print(f"Saved FIM for {baseline_name} to {save_path}")


# =========================
#  Running all baselines
# =========================

def reset_weights(model: nn.Module, initial_state_dict: dict) -> nn.Module:
    model.load_state_dict(initial_state_dict, strict=True)
    for p in model.parameters():
        p.grad = None
    return model

def run_all_baselines(sparsity: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sparsity = float(max(0.0, min(1.0, sparsity)))
    print(f"\n>>> Using sparsity = {sparsity:.3f} (fraction of weights/filters pruned)\n")

    train_loader, test_loader, fim_loader = get_mnist_loaders(batch_size=128,
                                                              fim_subset_size=1024)
    results = {}

    epochs = 20
    lr = 1e-3
    #fim_layers = ['fc2.weight']

    # Dense
    print("\n=== Baseline: Dense (no pruning) ===")
    dense_model = SimpleConvNet()
    initial_state_dict = copy.deepcopy(dense_model.state_dict())
    dense_model, dense_opt, dense_crit = train_model(dense_model, train_loader,
                                                     epochs=epochs, lr=lr,
                                                     device=device)
    acc_dense = evaluate(dense_model, test_loader, device)
    print(f"Dense accuracy: {acc_dense:.4f}")

    fim_dense = FisherInformationMatrix(dense_model,
                                        dense_crit,
                                        dense_opt,
                                        fim_loader,
                                        complete_fim=True)
    print("Dense FIM logdet:", fim_dense.logdet_ratio)
    save_fim(fim_dense, "dense")

    results['dense'] = {
        'accuracy': acc_dense,
        'logdet': fim_dense.logdet,
        'diaglogdet': fim_dense.diaglogdet,
        'logdet_ratio': fim_dense.logdet_ratio
    }

    # OBD
    print("\n=== Baseline: OBD (post-training) ===")
    obd_model = SimpleConvNet()
    obd_model = reset_weights(obd_model, initial_state_dict)
    obd_model, obd_opt, obd_crit = train_model(obd_model, train_loader,
                                               epochs=epochs, lr=lr,
                                               device=device)
    obd_model = obd_prune(obd_model, train_loader, obd_crit,
                          sparsity=sparsity, num_batches=10, device=device)
    acc_obd = evaluate(obd_model, test_loader, device)
    print(f"OBD accuracy after pruning: {acc_obd:.4f}")

    fim_obd = FisherInformationMatrix(obd_model,
                                      obd_crit,
                                      obd_opt,
                                      fim_loader,
                                      complete_fim=True)
    print("OBD FIM logdet:", fim_obd.logdet_ratio)
    save_fim(fim_obd, "obd")

    results['obd'] = {
        'accuracy': acc_obd,
        'logdet': fim_obd.logdet,
        'diaglogdet': fim_obd.diaglogdet,
        'logdet_ratio': fim_obd.logdet_ratio
    }

    # Structured filter pruning
    print("\n=== Baseline: Structured filter pruning (Li et al.) ===")
    filt_model = SimpleConvNet()
    filt_model = reset_weights(filt_model, initial_state_dict)
    filt_model, filt_opt, filt_crit = train_model(filt_model, train_loader,
                                                  epochs=epochs, lr=lr,
                                                  device=device)
    filt_model = structured_filter_prune(filt_model, amount_per_layer=sparsity)
    acc_filt = evaluate(filt_model, test_loader, device)
    print(f"Filter-pruned accuracy: {acc_filt:.4f}")

    fim_filt = FisherInformationMatrix(filt_model,
                                       filt_crit,
                                       filt_opt,
                                       fim_loader,
                                       complete_fim=True)
    print("Filter pruning FIM logdet:", fim_filt.logdet_ratio)
    save_fim(fim_filt, "filter_pruning")

    results['filter_pruning'] = {
        'accuracy': acc_filt,
        'logdet': fim_filt.logdet,
        'diaglogdet': fim_filt.diaglogdet,
        'logdet_ratio': fim_filt.logdet_ratio
    }

    # SNIP
    print("\n=== Baseline: SNIP (prune at init) ===")
    snip_model = SimpleConvNet()
    snip_model = reset_weights(snip_model, initial_state_dict)
    snip_model.to(device)
    snip_crit = nn.CrossEntropyLoss()

    snip_model = snip_prune(snip_model,
                            train_loader,
                            snip_crit,
                            sparsity=sparsity,
                            num_batches=1,
                            device=device)

    snip_opt = torch.optim.Adam(snip_model.parameters(), lr=lr)
    snip_model, snip_opt, snip_crit = train_model(snip_model, train_loader,
                                                  epochs=epochs, lr=lr,
                                                  device=device)
    acc_snip = evaluate(snip_model, test_loader, device)
    print(f"SNIP accuracy after training: {acc_snip:.4f}")

    fim_snip = FisherInformationMatrix(snip_model,
                                       snip_crit,
                                       snip_opt,
                                       fim_loader,
                                       complete_fim=True)
    print("SNIP FIM logdet:", fim_snip.logdet_ratio)
    save_fim(fim_snip, "snip")

    results['snip'] = {
        'accuracy': acc_snip,
        'logdet': fim_snip.logdet,
        'diaglogdet': fim_snip.diaglogdet,
        'logdet_ratio': fim_snip.logdet_ratio
    }

    print("\n=== Summary ===")
    for name, res in results.items():
        print(f"{name}: acc={res['accuracy']:.4f}, "
              f"logdet={res['logdet']:.3f}, "
              f"diaglogdet={res['diaglogdet']:.3f}, "
              f"logdet_ratio={res['logdet_ratio']:.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.9,
        help="Fraction of parameters/filters to prune (0.0–1.0).",
    )
    args = parser.parse_args()

    run_all_baselines(sparsity=args.sparsity)

