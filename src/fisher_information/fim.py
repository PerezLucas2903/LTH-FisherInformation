import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FisherInformationMatrix:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim, dataloader: torch.utils.data.DataLoader, 
                 complete_fim: bool = True, layers: list = None, mask: dict = None, sampling_type: str = 'complete', 
                 sampling_frequency: tuple = None):
        
        """ Fisher information Matrix computation class.
        params:
            model: nn.Module
            criterion: nn.Module
            optimizer: torch.optim
            dataloader: torch.utils.data.DataLoader witch batch size = 1 to compute gradients per sample
            complete_fim: bool, if True computes the complete FIM, else layer-wise FIM
            layers: list of layer names to compute FIM for (if complete_fim is False)
            mask: dict of masks to apply to gradients
            sampling_type: str, one of ['complete', 'x_in_x', 'x_skip_y']
            sampling_frequency: tuple, frequency for sampling gradients (only for 'x_in_x' and 'x_skip_y' sampling types). Example: (2,) for 'x_in_x' means take every 2nd gradient; (2,3) for 'x_skip_y' means take 2 gradients, skip 3 gradients in a cycle.

        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.layers = layers
        self.complete_fim = complete_fim
        self.mask = mask
        self.sampling_type = sampling_type
        self.sampling_frequency = sampling_frequency

        if self.layers is None and not self.complete_fim:
            raise ValueError("Either 'layers' must be specified or 'complete_fim' must be True.")
        
        if self.layers is not None:
            if not isinstance(self.layers, list):
                raise ValueError("'layers' must be a list of layer names.")
            param_names = [n for n, p in model.named_parameters()]
            for name in self.layers:
                if name not in param_names:
                    raise ValueError(f"Layer name '{name}' not found in model parameters.")
        
        self.sampling_masks = self._make_sampling_masks(model)
        self.concat_mask = self._make_concat_mask(model)

        
        if complete_fim:
            n_params = self.concat_mask.sum().item() if self.concat_mask is not None else sum(p.numel() for p in model.parameters() if p.requires_grad)
            #n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.fim = {'complete': torch.zeros((n_params, n_params), device=self.device)}
        else:
            self.fim = {name: torch.zeros((mask.sum(), mask.sum()), device=self.device) for name, mask in self.sampling_masks.items() if name in self.layers}

        self.compute_fim(model.to(self.device), dataloader, optimizer)
        self.compute_logdet_metrics()

    def _fim_to_device(self):
        for key in self.fim:
            self.fim[key] = self.fim[key].to(self.device)
    
    def _fim_to_cpu(self):
        for key in self.fim:
            self.fim[key] = self.fim[key].to('cpu')

    def _make_sampling_masks(self, model):
        if self.complete_fim:
            return None
    
        elif self.sampling_type == 'complete' and not self.complete_fim:
            sampling_masks = {}
            for name, param in model.named_parameters():
                if name in self.layers:
                    sampling_masks[name] = torch.ones(param.numel()).to(bool).to(self.device)

            return sampling_masks
        
        elif self.sampling_type == 'x_in_x':
            sampling_masks = {}
            for name, param in model.named_parameters():
                if name in self.layers:
                    mask = [True if i % self.sampling_frequency[0] == 0 else False for i in range(0, param.numel())]
                    sampling_masks[name] = torch.Tensor(mask, device=self.device).to(bool)
                if self.mask is not None and name in self.mask:
                    pruning_mask = self.mask[name].view(-1).to(dtype=torch.bool, device=self.device)
                    sampling_masks[name] = sampling_masks[name] & pruning_mask
            return sampling_masks
        
        elif self.sampling_type == 'x_skip_y':
            sampling_masks = {}
            cycle_length = sum(self.sampling_frequency)
            for name, param in model.named_parameters():
                if name in self.layers:
                    mask = (torch.arange(param.numel(), device=self.device) % cycle_length) < self.sampling_frequency[0]
                    #sampling_masks[name] = torch.Tensor(mask, device=self.device).to(bool)
                    sampling_masks[name] = mask.to(bool)
                    if self.mask is not None and name in self.mask:
                        pruning_mask = self.mask[name].view(-1).to(dtype=torch.bool, device=self.device)
                        sampling_masks[name] = sampling_masks[name] & pruning_mask
            return sampling_masks
        
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_type}")
        
    def _make_concat_mask(self, model):
        """ Mask is a dict of each of the models layer. In order to calculate logdet metrics of the complete FIM, we need to concatenate the masks into a single mask.
        This is only needed if a mask is provided AND complete FIM is being computed.

        returns: concat_mask: torch.Tensor
        """
        if self.complete_fim is False:
            return None
        
        if self.mask is None:
            return torch.ones(sum(p.numel() for p in model.parameters() if p.requires_grad)).to(self.device).bool()
        
        concat_mask = torch.Tensor().to(self.device)
        for name, param in model.named_parameters():
            if name in self.mask.keys():
                concat_mask = torch.cat( (concat_mask, self.mask[name].view(-1).to(self.device)) )
            else:
                concat_mask = torch.cat( (concat_mask, torch.ones(param.numel()).to(self.device)) )

        return concat_mask.to(bool)

    def _compute_fim_complete(self, model, dataloader, optimizer) -> dict:
        model.eval()
        eps = 1e-8
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in self.mask.keys():
                            param.grad *= self.mask[name]

            grad = torch.cat( list( p.grad.detach().view(-1) for p in model.parameters() ) ).view(-1)
            self.fim['complete'] +=  torch.outer(grad[self.concat_mask], grad[self.concat_mask])


        self.fim['complete'] = torch.divide(self.fim['complete'], len(dataloader.dataset))
        self.fim['complete'] += eps * torch.eye(self.fim['complete'].shape[0], dtype=self.fim['complete'].dtype).to(self.device)

    def _compute_fim_layerwise(self, model, dataloader, optimizer) -> dict:
        model.eval()
        eps = 1e-8
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
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
                    grad = param.grad.view(-1).detach()[self.sampling_masks[name]]
                    self.fim[name] += torch.outer(grad, grad)

        for name in self.fim:
            self.fim[name] = torch.divide(self.fim[name], len(dataloader.dataset))
            n = self.fim[name].shape[0]
            idx = torch.arange(n, device=self.fim[name].device)
            self.fim[name][idx, idx] += eps

            #self.fim[name] += eps * torch.eye(self.fim[name].shape[0], dtype=self.fim[name].dtype).to(self.device)

    def _to_correlation_single(self, key, eps):
        d = torch.diagonal(self.fim[key], dim1=-2, dim2=-1)  # (..., n)

        if eps > 0:
            d = d + eps
        inv_sqrt_d = 1.0 / torch.sqrt(d)
        # constrói D^{-1/2} como diagonal batelada
        Dinv2 = torch.diag_embed(inv_sqrt_d)     # (..., n, n)
        self.corr_fim[key] = Dinv2 @ self.fim[key] @ Dinv2
        self.corr_fim_logdet_ratio[key] = 1 - torch.slogdet(self.corr_fim[key])[-1].item()

    def to_correlation(self):
        """
        A: (..., n, n) SPD (real simétrica).
        Retorna C = D^{-1/2} A D^{-1/2} com diag(C)=1.
        eps: opcional p/ estabilidade (raramente necessário em SPD bem condicionado).
        """
        eps = 0 #1e-8
        self.corr_fim = {}
        self.corr_fim_logdet_ratio = {}
        for key in self.fim:
            self._to_correlation_single(key, eps)

    def compute_fim(self, model, dataloader, optimizer) -> dict:
        if self.complete_fim:
            self._compute_fim_complete(model, dataloader, optimizer)
        else:
            self._compute_fim_layerwise(model, dataloader, optimizer)

    def compute_logdet_metrics(self):
        if self.complete_fim:
            fim = self.fim['complete']
            # if self.concat_mask is not None:
            #     fim = fim[self.concat_mask][:, self.concat_mask]
            self.logdet = torch.slogdet(fim)[-1].item()
            self.diaglogdet = torch.sum(torch.log(torch.diag(fim))).item()
            self.logdet_ratio =  (self.diaglogdet - self.logdet)/2
            self.logdet_ratio_per_dim = self.logdet_ratio / fim.shape[0] 
        else:
            logdets = {}
            diaglogdets = {}
            logdet_ratios = {}
            logdet_ratios_per_dim = {}
            for name, fim in self.fim.items():
                logdet = torch.slogdet(fim)[-1].item()
                diaglogdet = torch.sum(torch.log(torch.diag(fim))).item()
                logdet_ratio = (diaglogdet - logdet)/2

                logdets[name] = logdet
                diaglogdets[name] = diaglogdet
                logdet_ratios[name] = logdet_ratio
                logdet_ratios_per_dim[name] = logdet_ratio / fim.shape[0]

            self.logdet = logdets
            self.diaglogdet = diaglogdets
            self.logdet_ratio = logdet_ratios
            self.logdet_ratio_per_dim = logdet_ratios_per_dim