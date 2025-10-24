import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FisherInformationMatrix:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim,complete_fim: bool = True, layers: list = None, mask: dict = None, sampling_type: str = 'complete', sampling_frequency: tuple = None):
        """ Fisher information Matrix computation class.
        params:
            model: nn.Module
            criterion: nn.Module
            optimizer: torch.optim
            complete_fim: bool, if True computes the complete FIM, else layer-wise FIM
            layers: list of layer names to compute FIM for (if complete_fim is False)
            mask: dict of masks to apply to gradients
            sampling_type: str, one of ['complete', 'x_in_x', 'x_skip_y']
            sampling_frequency: tuple, frequency for sampling gradients (only for 'x_in_x' and 'x_skip_y' sampling types). Example: (2,) for 'x_in_x' means take every 2nd gradient; (2,3) for 'x_skip_y' means take 2 gradients, skip 3 gradients in a cycle.

        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.criterion = criterion
        self.layers = layers
        self.complete_fim = complete_fim
        self.optimizer = optimizer
        self.mask = mask
        self.sampling_type = sampling_type
        self.sampling_frequency = sampling_frequency

        if self.layers is None and not self.complete_fim:
            raise ValueError("Either 'layers' must be specified or 'complete_fim' must be True.")
        
        self.sampling_masks = self._make_sampling_masks()
        self.concat_mask = self._make_concat_mask()
        self.logdet = None
        self.diaglogdet = None
        self.logdet_ratio = None
        
        if complete_fim:
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.fim = {'complete': torch.zeros((n_params, n_params), device=self.device)}
        else:
            self.fim = {name: torch.zeros((param.numel(), param.numel()), device=self.device) for name, param in self.model.named_parameters() if param.requires_grad and name in self.layers}

    def _make_sampling_masks(self):
        if self.sampling_type == 'complete':
            return None
        
        elif self.sampling_type == 'x_in_x':
            sampling_masks = {}
            for name, param in self.model.named_parameters():
                if name == self.layers:
                    mask = [True if i % self.sampling_frequency[0] == 0 else False for i in range(0, param.numel())]
                    sampling_masks[name] = torch.Tensor(mask).to(bool).to(self.device)
            return sampling_masks
        
        elif self.sampling_type == 'x_skip_y':
            sampling_masks = {}
            cycle_length = sum(self.sampling_frequency)
            for name, param in self.model.named_parameters():
                if name == self.layers:
                    mask = (torch.arange(param.numel()) % cycle_length) < self.sampling_frequency[0]
                    sampling_masks[name] = torch.Tensor(mask).to(bool).to(self.device)
            return sampling_masks
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")
        
    def _make_concat_mask(self):
        """ Mask is a dict of each of the models layer. In order to calculate logdet metrics of the complete FIM, we need to concatenate the masks into a single mask.
        This is only needed if a mask is provided AND complete FIM is being computed.

        returns: concat_mask: torch.Tensor
        """
        if self.complete_fim is False:
            return None
        
        if self.mask is None:
            return torch.ones(sum(p.numel() for p in self.model.parameters() if p.requires_grad)).to(self.device).bool()
        
        concat_mask = torch.Tensor().to(self.device)
        for name, param in self.model.named_parameters():
            if name in self.mask.keys():
                concat_mask = torch.cat( (concat_mask, self.mask[name].view(-1).to(self.device)) )
            else:
                concat_mask = torch.cat( (concat_mask, torch.ones(param.numel()).to(self.device)) )

        return concat_mask.to(bool)

    def _compute_fim_complete(self, data_loader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        eps = 1e-8

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.mask is not None:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.mask.keys():
                            param.grad *= self.mask[name]

            grad = torch.cat( list( p.grad.view(-1) for p in self.model.parameters() ) ).view(-1, 1)
            self.fim['complete'] +=  torch.outer(grad, grad)


        self.fim['complete'] = torch.divide(self.fim['complete'], len(data_loader.dataset))
        self.fim['complete'] += eps * torch.eye(self.fim['complete'].shape[0], dtype=self.fim['complete'].dtype).to(self.device)

    def _compute_fim_layerwise(self, data_loader: torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        eps = 1e-8

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            if self.mask is not None:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in self.mask.keys():
                            param.grad *= self.mask[name]

            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.layers:
                    grad = param.grad.view(-1, 1)
                    self.fim[name] += torch.outer(grad, grad)

        for name in self.fim:
            self.fim[name] = torch.divide(self.fim[name], len(data_loader.dataset))
            self.fim[name] += eps * torch.eye(self.fim[name].shape[0], dtype=self.fim[name].dtype).to(self.device)

    def compute_fim(self, data_loader: torch.utils.data.DataLoader) -> dict:
        if self.complete_fim:
            self._compute_fim_complete(data_loader)
        else:
            self._compute_fim_layerwise(data_loader)

    def compute_logdet_metrics(self):
        if self.complete_fim:
            fim = self.fim['complete']
            if self.concat_mask is not None:
                fim = fim[self.concat_mask][:, self.concat_mask]
            eigvals, eigvecs = torch.linalg.eig(fim)
            eigvals = eigvals.to(torch.float)
            #eigvals = eigvals[eigvals > 0] # If fim is positive definite, this is not needed
            self.logdet = torch.sum(torch.log(eigvals)).item()
            self.diaglogdet = torch.sum(torch.log(torch.diag(fim))).item()
            self.logdet_ratio = self.logdet - self.diaglogdet
        else:
            logdets = {}
            diaglogdets = {}
            logdet_ratios = {}
            for name, fim in self.fim.items():
                eigvals, eigvecs = torch.linalg.eig(fim)
                #eigvals = eigvals[eigvals > 0] # If fim is positive definite, this is not needed
                eigvals = eigvals.to(torch.float)
                logdet = torch.sum(torch.log(eigvals)).item()
                diaglogdet = torch.sum(torch.log(torch.diag(fim))).item()
                logdet_ratio = logdet - diaglogdet

                logdets[name] = logdet
                diaglogdets[name] = diaglogdet
                logdet_ratios[name] = logdet_ratio

            self.logdet = logdets
            self.diaglogdet = diaglogdets
            self.logdet_ratio = logdet_ratios