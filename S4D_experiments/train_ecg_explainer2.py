import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F

# ===================================================================
# >> CONFIGURACIÓN DE RUTAS <<
# ===================================================================
PATH_A_TU_PROYECTO_S4D_ECG = '../../S4D-ECG' 
PATH_A_TU_MODELO_S4D_ENTRENADO = '../../S4D-ECG/s4_results/S4D/model.pt'
PATH_A_TUS_DATOS = '../../S4D-ECG/s4_results/embeddings/ecg_embeddings_testset.pt'
# ===================================================================

sys.path.append(PATH_A_TU_PROYECTO_S4D_ECG)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.models.s4.s4d import S4D
from Train import dropout_fn
from txai.models.mask_generators.maskgen import MaskGenerator
# --- ¡IMPORTACIONES CLAVE PARA LAS PÉRDIDAS DEL PAPER! ---
from txai.utils.predictors.loss import GSATLoss, ConnectLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# >> CLASES (S4Model, S4D_Explainer, Poly1BCELoss) <<
# -----------------------------------------------------------------------------
class S4Model(nn.Module):
    def __init__(self, d_input, d_output=8, d_model=128, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
        super().__init__()
        self.prenorm = prenorm; self.encoder = nn.Linear(d_input, d_model); self.s4_layers = nn.ModuleList(); self.norms = nn.ModuleList(); self.dropouts = nn.ModuleList()
        for _ in range(n_layers): self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))); self.norms.append(nn.LayerNorm(d_model)); self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Linear(d_model, d_output)
    def forward(self, x, times, **kwargs):
        x = self.encoder(x); x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm: z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z); z = dropout(z); x = z + x
            if not self.prenorm: x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2); embedding_secuencia = x; pooled_output = x.mean(dim=1); final_prediction = self.decoder(pooled_output); final_prediction = nn.functional.sigmoid(final_prediction)
        return final_prediction, embedding_secuencia, pooled_output

class S4D_Explainer(nn.Module):
    def __init__(self, s4_model, d_model, max_len, d_inp):
        super().__init__()
        self.s4_model = s4_model
        self.mask_generator = MaskGenerator(d_z=d_model, max_len=max_len, d_inp=d_inp)
    def forward(self, x, times):
        self.s4_model.eval()
        with torch.no_grad():
            prediction, embedding, pooled_embedding = self.s4_model(x, times)
        embedding_T = embedding.transpose(0, 1); times_T = times.transpose(0, 1)
        mask_soft, mask_hard = self.mask_generator(z_seq=embedding_T, src=embedding_T, times=times_T)
        return {'prediction': prediction, 'mask_soft': mask_soft, 'mask_hard': mask_hard, 'pooled_embedding': pooled_embedding}

class Poly1BCELoss(nn.Module):
    def __init__(self, epsilon=2.0, reduction='mean'):
        super().__init__(); self.epsilon = epsilon; self.reduction = reduction
    def forward(self, logits, labels):
        if torch.all((logits >= 0) & (logits <= 1)): logits = torch.logit(logits, eps=1e-6)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        pt = torch.exp(-bce_loss); poly1_loss = bce_loss + self.epsilon * (1 - pt)
        if self.reduction == 'mean': return poly1_loss.mean()
        else: return poly1_loss

def train_explainer(explainer, dataloader, optimizer, criterions, betas, device):
    explainer.train()
    pbar = tqdm(dataloader, desc="Entrenando (GSAT+Connect)")
    for x, times, y, _ in dataloader:
        x, times, y = x.to(device), times.to(device), y.to(device)
        optimizer.zero_grad()
        output = explainer(x, times)
        
        mask_soft = output['mask_soft']
        mask_hard = output['mask_hard']
        
        # --- Cálculo de las pérdidas del paper ---
        gsat_loss = criterions['gsat'](mask_soft)
        connect_loss = criterions['connect'](mask_hard)
        
        # --- Pérdida de Fidelidad ---
        prediction_after_masking, _, _ = explainer.s4_model(x * mask_hard.detach(), times)
        fidelity_loss = criterions['fidelity'](prediction_after_masking, y)
        
        # --- Combinamos las pérdidas con sus betas ---
        loss = (betas['fidelity'] * fidelity_loss) + (betas['gsat'] * gsat_loss) + (betas['connect'] * connect_loss)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': loss.item(), 'fidelidad': fidelity_loss.item(), 'gsat': gsat_loss.item(), 'connect': connect_loss.item()})

# ===================================================================
# >> EJECUCIÓN PRINCIPAL <<
# ===================================================================
if __name__ == '__main__':
    
    # --- PANEL DE CONTROL DE HIPERPARÁMETROS ---
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    BETA_FIDELITY = 1.0
    BETA_GSAT = 0.5      # Controla la dispersión (más alto = más dispersa)
    BETA_CONNECT = 1.0   # Controla la continuidad
    POLY1_EPSILON = 2.0
    GSAT_R = 0.5         # Sparsity regularizer for GSAT
    # ----------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    print("==> Cargando datos de ECG...")
    data_dict = torch.load(PATH_A_TUS_DATOS)
    original_signals, labels = data_dict['original_signals'], data_dict['labels']
    d_inp, max_len, n_classes = original_signals.shape[2], original_signals.shape[1], labels.shape[1]
    
    times = torch.arange(max_len).unsqueeze(0).repeat(original_signals.shape[0], 1)
    indices = torch.arange(original_signals.shape[0])
    dataset = TensorDataset(original_signals, times, labels, indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"==> Cargando tu modelo S4D-ECG...")
    s4_model_trained = S4Model(d_input=d_inp, d_output=n_classes, d_model=128, n_layers=4)
    s4_model_trained.load_state_dict(torch.load(PATH_A_TU_MODELO_S4D_ENTRENADO, map_location=device))
    
    print("==> Construyendo el nuevo modelo explicador...")
    explainer = S4D_Explainer(s4_model=s4_model_trained, d_model=128, max_len=max_len, d_inp=d_inp)
    explainer.to(device)
    
    optimizer = torch.optim.AdamW(explainer.mask_generator.parameters(), lr=LEARNING_RATE)
    
    criterions = {
        'fidelity': Poly1BCELoss(epsilon=POLY1_EPSILON),
        'gsat': GSATLoss(r=GSAT_R),
        'connect': ConnectLoss()
    }
    
    betas = {'fidelity': BETA_FIDELITY, 'gsat': BETA_GSAT, 'connect': BETA_CONNECT}
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_explainer(explainer, dataloader, optimizer, criterions, betas, device)
    
    save_path = 'models/my_s4d_explainer_gsat_final.pt'
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(explainer.state_dict(), save_path)
    print(f"\n¡ÉXITO! Explicador entrenado y guardado en '{save_path}'")