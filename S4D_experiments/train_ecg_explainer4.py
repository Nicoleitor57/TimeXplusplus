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
# --- ¡IMPORTACIONES CLAVE DEL PAPER! ---
from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
from txai.trainers.train_mv6_consistency import train_mv6_consistency
from txai.utils.predictors.select_models import *

from txai.utils.predictors.loss_cl import LabelConsistencyLoss, EmbedConsistencyLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Poly1BCELoss(nn.Module):
    def __init__(self, epsilon=2.0, reduction='mean'):
        super().__init__(); self.epsilon = epsilon; self.reduction = reduction
    def forward(self, logits, labels):
        if torch.all((logits >= 0) & (logits <= 1)): logits = torch.logit(logits, eps=1e-6)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        pt = torch.exp(-bce_loss); poly1_loss = bce_loss + self.epsilon * (1 - pt)
        if self.reduction == 'mean': return poly1_loss.mean()
        else: return poly1_loss

# -----------------------------------------------------------------------------
# >> TU CLASE S4Model (DEBE SER IDÉNTICA A LA DE TU PROYECTO ORIGINAL) <<
# -----------------------------------------------------------------------------
class S4Model(nn.Module):
    def __init__(self, d_input, d_output=8, d_model=128, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
        super().__init__()
        self.prenorm = prenorm; self.encoder = nn.Linear(d_input, d_model); self.s4_layers = nn.ModuleList(); self.norms = nn.ModuleList(); self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))); self.norms.append(nn.LayerNorm(d_model)); self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Linear(d_model, d_output)
        
    def forward(self, x, times, **kwargs):
        x = self.encoder(x); x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm: z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z); z = dropout(z); x = z + x
            if not self.prenorm: x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        embedding_secuencia = x 
        pooled_output = x.mean(dim=1)
        final_prediction = self.decoder(pooled_output)
        final_prediction = nn.functional.sigmoid(final_prediction)
        
        return final_prediction, embedding_secuencia

# ===================================================================
# >> EJECUCIÓN PRINCIPAL <<
# ===================================================================
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- Carga de Datos ---
    print("==> Cargando datos de ECG...")
    data_dict = torch.load(PATH_A_TUS_DATOS)
    original_signals, labels = data_dict['original_signals'], data_dict['labels']
    d_inp, max_len, n_classes = original_signals.shape[2], original_signals.shape[1], labels.shape[1]
    
    times = torch.arange(max_len).unsqueeze(0).repeat(original_signals.shape[0], 1)
    indices = torch.arange(original_signals.shape[0])
    train_size = int(0.8 * len(original_signals))
    
    train_dataset = TensorDataset(original_signals[:train_size], times[:train_size], labels[:train_size], indices[:train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_tuple = (original_signals[train_size:], times[train_size:], labels[train_size:], indices[train_size:])

    # Creamos las tuplas de datos de entrenamiento y validación
    train_tuple = (original_signals[:train_size], times[:train_size], labels[:train_size])
    val_tuple = (original_signals[train_size:], times[train_size:], labels[train_size:], indices[train_size:])
    
    # Creamos el DataLoader para el entrenamiento
    train_dataset = TensorDataset(original_signals[:train_size], times[:train_size], labels[:train_size], indices[:train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # --- Construcción del Modelo Explicador TimeXModel ---
    print("==> Construyendo el modelo explicador TimeX...")
    mu = original_signals.mean(dim=0); std = original_signals.std(unbiased=True, dim=0)
    
    # TimeXModel necesita saber que el 'archtype' (arquitectura) NO es 'transformer'
    # para que su lógica interna se ajuste. Le decimos que es 's4' (un nombre inventado).
    abl_params = AblationParameters(archtype='s4')

    # # Creamos la instancia de TimeXModel. Por dentro, creará un encoder que no usaremos.
    # explainer = TimeXModel(
    #     d_inp=d_inp,
    #     max_len=max_len,
    #     n_classes=n_classes,
    #     n_prototypes=50,
    #     gsat_r=0.5,
    #     ablation_parameters=abl_params,
    #     masktoken_stats=(mu, std)
    # )

    # --- ¡CAMBIOS CLAVE AQUÍ! ---
    # 1. Recreamos el diccionario de configuración por defecto
    targs = transformer_default_args.copy() # Usamos una copia
    # Forzamos la dimensión interna para que coincida con tu S4D
    targs['d_model'] = 128 

    # 2. Definimos el diccionario para activar las pérdidas internas
    loss_weight_dict = {
        'gsat': 0.5, # Puedes ajustar este peso
        'connect': 1.0 # Puedes ajustar este peso
    }

    # 3. Llamamos a TimeXModel con TODOS los argumentos necesarios
    explainer = TimeXModel(
        d_inp=d_inp,
        max_len=max_len,
        n_classes=n_classes,
        n_prototypes=50,
        gsat_r=0.5,
        transformer_args=targs, # <-- AÑADIDO
        ablation_parameters=abl_params,
        loss_weight_dict=loss_weight_dict, # <-- AÑADIDO
        masktoken_stats=(mu, std)
    )

    # --- "Inyección" de tu S4D-ECG ---
    print("==> Reemplazando el encoder interno con tu S4D-ECG entrenado...")
    s4_model_trained = S4Model(d_input=d_inp, d_output=n_classes, d_model=128, n_layers=4)
    s4_model_trained.load_state_dict(torch.load(PATH_A_TU_MODELO_S4D_ENTRENADO, map_location=device))
    
    # ¡Este es el paso clave! Reemplazamos el motor.
    explainer.encoder_main = s4_model_trained
    explainer.to(device)

    # Congelamos los pesos de tu S4D-ECG.
    for param in explainer.encoder_main.parameters():
        param.requires_grad = False

    # --- Entrenamiento del Explicador ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, explainer.parameters()), lr=1e-3)
    
    # Creamos los criterios de pérdida necesarios
    clf_criterion = Poly1BCELoss(epsilon=2.0)
    sim_criterion_label = LabelConsistencyLoss()
    sim_criterion_cons = EmbedConsistencyLoss()
    sim_criterion = [sim_criterion_cons, sim_criterion_label]
    selection_criterion = simloss_on_val_wboth(sim_criterion, lam=1.0)
    
    epochs = 30
    
    # Usamos la función de entrenamiento original de la librería
    best_model = train_mv6_consistency(
        explainer,
        optimizer = optimizer,
        train_loader = train_loader,
        val_tuple = val_tuple[:3], # Pasamos solo X, times, y
        train_tuple = train_tuple,
        clf_criterion = clf_criterion,
        sim_criterion = sim_criterion,
        selection_criterion = selection_criterion,
        # Hiperparámetros
        beta_exp = 1.0, # Peso para GSAT + Connect
        beta_sim = 1.0, # Peso para Consistencia
        num_epochs = epochs,
        early_stopping = True,
        save_path = 'models/my_s4d_explainer_timex_final.pt',
        # Argumentos adicionales
        label_matching = True,
        embedding_matching = True,
        use_scheduler = True,
        #device = device
    )
    
    print(f"\n¡ÉXITO! Explicador entrenado y guardado.")