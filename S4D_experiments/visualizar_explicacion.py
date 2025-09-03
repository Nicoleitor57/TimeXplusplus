# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import os

# # ===================================================================
# # >> CONFIGURACIÓN DE RUTAS Y PARÁMETROS (COMPLETA ESTAS LÍNEAS) <<
# # ===================================================================
# # --- RUTAS DE LOS MODELOS ---
# PATH_A_TU_PROYECTO_S4D_ECG = '../../S4D-ECG' 
# PATH_A_TU_MODELO_S4D_ENTRENADO = '../../S4D-ECG/s4_results/S4D/model.pt'
# PATH_A_TU_EXPLICADOR_ENTRENADO = 'models/my_s4d_explainer_smooth.pt' # El explicador que entrenaste

# # --- RUTA AL ECG QUE QUIERES EXPLICAR ---
# PATH_A_UN_ECG_DAT = '../../S4D-ECG/exams/19009_lr.dat' # <<-- ¡AQUÍ PONES EL .dat A EXPLICAR!

# # --- PARÁMETROS DE LA SEÑAL ---
# NUM_LEADS = 12
# SAMPLING_HZ = 100 # La frecuencia REAL de tus archivos .dat
# DURATION_S = 10
# # ===================================================================

# # Añadimos las rutas necesarias para importar las clases
# sys.path.append(PATH_A_TU_PROYECTO_S4D_ECG)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# # Importamos los componentes necesarios
# from src.models.s4.s4d import S4D
# from Train import dropout_fn
# from txai.models.mask_generators.maskgen import MaskGenerator

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# # -----------------------------------------------------------------------------
# # >> CLASES DE MODELOS (COPIADAS AQUÍ PARA SER AUTOCONTENIDO) <<
# # -----------------------------------------------------------------------------
# class S4Model(nn.Module):
#     def __init__(self, d_input, d_output=10, d_model=128, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
#         super().__init__()
#         self.prenorm = prenorm
#         self.encoder = nn.Linear(d_input, d_model)
#         self.s4_layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.dropouts = nn.ModuleList()
#         for _ in range(n_layers):
#             self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
#             self.norms.append(nn.LayerNorm(d_model))
#             self.dropouts.append(dropout_fn(dropout))
#         self.decoder = nn.Linear(d_model, d_output)

#     def forward(self, x, times, **kwargs):
#         x = self.encoder(x)
#         x = x.transpose(-1, -2)
#         for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
#             z = x
#             if self.prenorm:
#                 z = norm(z.transpose(-1, -2)).transpose(-1, -2)
#             z, _ = layer(z)
#             z = dropout(z)
#             x = z + x
#             if not self.prenorm:
#                 x = norm(x.transpose(-1, -2)).transpose(-1, -2)
#         x = x.transpose(-1, -2)
#         embedding_secuencia = x 
#         pooled_output = x.mean(dim=1)
#         final_prediction = self.decoder(pooled_output)
#         final_prediction = nn.functional.sigmoid(final_prediction)
#         return final_prediction, embedding_secuencia
        
# class S4D_Explainer(nn.Module):
#     def __init__(self, s4_model, d_model, max_len):
#         super().__init__()
#         self.s4_model = s4_model
#         self.mask_generator = MaskGenerator(d_z=d_model, max_len=max_len, d_inp=12)

#     def forward(self, x, times):
#         self.s4_model.eval()
#         with torch.no_grad():
#             prediction, embedding = self.s4_model(x, times)
        
#         embedding_T = embedding.transpose(0, 1)
#         times_T = times.transpose(0, 1)
#         mask_tuple = self.mask_generator(z_seq=embedding_T, src=embedding_T, times=times_T)
#         mask = mask_tuple[1]
        
#         return {'prediction': prediction, 'mask': mask}

# # --- FUNCIÓN PARA CARGAR Y PROCESAR EL ECG ---
# def load_ecg_signal(filepath, num_leads, sampling_hz, duration_s):
#     print(f"==> Cargando señal de {sampling_hz}Hz desde '{filepath}'...")
#     # Se asume un tipo de dato int16, común en ECG. Cambia si es diferente.
#     signal_flat = np.fromfile(filepath, dtype=np.int16)
    
#     expected_points = sampling_hz * duration_s * num_leads
#     sequence_length = sampling_hz * duration_s
    
#     # Darle la forma correcta (Tiempo, Derivaciones)
#     signal_reshaped = signal_flat[:expected_points].reshape(sequence_length, num_leads)
    
#     # Convertir a Tensor de PyTorch
#     signal_tensor = torch.from_numpy(signal_reshaped.copy()).float()
    
#     print(f"Señal cargada. Forma final: {signal_tensor.shape}")
#     return signal_tensor

# # -----------------------------------------------------------------------------
# # >> EJECUCIÓN PRINCIPAL PARA VISUALIZACIÓN <<
# # -----------------------------------------------------------------------------
# if __name__ == '__main__':
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Usando dispositivo: {device}")

#     # --- Carga de la señal de ECG individual ---
#     signal_to_explain = load_ecg_signal(
#         filepath=PATH_A_UN_ECG_DAT,
#         num_leads=NUM_LEADS,
#         sampling_hz=SAMPLING_HZ,
#         duration_s=DURATION_S
#     )
#     max_len = signal_to_explain.shape[0]
#     d_inp = signal_to_explain.shape[1]
#     n_classes = 8 # El número de clases de tu modelo S4D

#     # --- Construcción y Carga de Modelos ---
#     print(f"==> Cargando tu modelo S4D-ECG entrenado...")
#     s4_model_trained = S4Model(d_input=d_inp, d_output=n_classes, d_model=128, n_layers=4)
#     s4_model_trained.load_state_dict(torch.load(PATH_A_TU_MODELO_S4D_ENTRENADO, map_location=device))
    
#     print(f"==> Cargando tu modelo Explicador entrenado...")
#     explainer = S4D_Explainer(s4_model=s4_model_trained, d_model=128, max_len=max_len)
#     explainer.load_state_dict(torch.load(PATH_A_TU_EXPLICADOR_ENTRENADO, map_location=device))
#     explainer.to(device)
#     explainer.eval()

#     # --- Generación de la Explicación ---
#     print(f"\nGenerando explicación para la señal...")
#     signal_batch = signal_to_explain.unsqueeze(0).to(device)
#     time_batch = torch.arange(max_len).unsqueeze(0).to(device).float()

#     output = explainer(signal_batch, time_batch)
#     prediction = output['prediction']
#     saliency_map = output['mask']
#     predicted_label_idx = torch.mode(torch.argmax(prediction, dim=-1)).values.item()

#     sample_np = signal_batch.cpu().detach().numpy().squeeze()
#     saliency_np = saliency_map.cpu().detach().numpy().squeeze()

#     # --- Graficar el Resultado ---
#     print(f"Predicción general del modelo para esta señal: Clase {predicted_label_idx}")
    
#     lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
#     fig, axes = plt.subplots(12, 1, figsize=(20, 15), sharex=True)
#     fig.suptitle(f"Explicación para Señal de ECG - Predicción: Clase {predicted_label_idx}", fontsize=16)

#     for i in range(NUM_LEADS):
#         ax = axes[i]
#         signal_lead = sample_np[:, i]
#         saliency_lead = saliency_np[:, i]
        
#         ax.plot(signal_lead, color='dodgerblue', linewidth=1.5)
        
#         time_steps = np.arange(len(signal_lead))
#         ax.fill_between(time_steps, np.min(signal_lead), np.max(signal_lead), 
#                         where=saliency_lead > 0.5,
#                         color='tomato', alpha=0.5)
        
#         ax.set_ylabel(lead_names[i])
#         ax.grid(True, linestyle='--', alpha=0.5)

#     axes[-1].set_xlabel(f"Pasos de Tiempo (a {SAMPLING_HZ} Hz)")
#     fig.legend(['Señal Original', 'Zona Importante'], loc='upper right')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
#     output_filename = "explicacion_ecg_final.png"
#     plt.savefig(output_filename)

#     print(f"\n¡ÉXITO! Gráfico de 12 derivaciones guardado como '{output_filename}'!")

import sys
import os
import argparse

# Agrega el directorio padre al Python Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../kardia_s4d/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt, sys, os
from src.models.s4.s4d import S4D
from Train import dropout_fn
from txai.models.mask_generators.maskgen import MaskGenerator
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# (Pega aquí tus clases S4Model y S4D_Explainer completas)
class S4Model(nn.Module):
    # ... tu clase S4Model completa ...
    def __init__(self, d_input, d_output=8, d_model=128, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
        super().__init__(); self.prenorm = prenorm; self.encoder = nn.Linear(d_input, d_model); self.s4_layers = nn.ModuleList(); self.norms = nn.ModuleList(); self.dropouts = nn.ModuleList()
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
        return final_prediction, embedding_secuencia

class S4D_Explainer(nn.Module):
    # ... tu clase S4D_Explainer completa ...
    def __init__(self, s4_model, d_model, max_len, d_inp):
        super().__init__()
        self.s4_model = s4_model
        self.mask_generator = MaskGenerator(d_z=d_model, max_len=max_len, d_inp=d_inp)
    def forward(self, x, times):
        self.s4_model.eval();
        with torch.no_grad(): prediction, embedding = self.s4_model(x, times)
        embedding_T = embedding.transpose(0, 1); times_T = times.transpose(0, 1)
        mask_tuple = self.mask_generator(z_seq=embedding_T, src=embedding_T, times=times_T); mask = mask_tuple[1]
        return {'prediction': prediction, 'mask': mask}

def load_ecg_signal(filepath, num_leads, sampling_hz, duration_s):
    signal_flat = np.fromfile(filepath, dtype=np.int16)
    expected_points = sampling_hz * duration_s * num_leads; sequence_length = sampling_hz * duration_s
    signal_reshaped = signal_flat[:expected_points].reshape(sequence_length, num_leads)
    return torch.from_numpy(signal_reshaped.copy()).float()

def generate_visual_explanation(config):
    # Esta función ahora toma una configuración y genera el gráfico
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    signal_to_explain = load_ecg_signal(config['ecg_path'], config['num_leads'], config['sampling_hz'], config['duration_s'])
    max_len, d_inp, n_classes = signal_to_explain.shape[0], signal_to_explain.shape[1], 8
    
    s4_model_trained = S4Model(d_input=d_inp, d_output=n_classes, d_model=128, n_layers=4)
    s4_model_trained.load_state_dict(torch.load(config['s4_model_path'], map_location=device))
    
    explainer = S4D_Explainer(s4_model=s4_model_trained, d_model=128, max_len=max_len, d_inp=d_inp)
    explainer.load_state_dict(torch.load(config['explainer_path'], map_location=device))
    explainer.to(device); explainer.eval()

    signal_batch = signal_to_explain.unsqueeze(0).to(device)
    time_batch = torch.arange(max_len).unsqueeze(0).to(device).float()

    output = explainer(signal_batch, time_batch)
    prediction, saliency_map = output['prediction'], output['mask']
    predicted_label_idx = torch.mode(torch.argmax(prediction, dim=-1)).values.item()

    sample_np = signal_batch.cpu().detach().numpy().squeeze()
    saliency_np = saliency_map.cpu().detach().numpy().squeeze()

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, axes = plt.subplots(12, 1, figsize=(20, 15), sharex=True)
    title = f"Explicación para '{os.path.basename(config['ecg_path'])}' con config '{config['run_name']}' - Pred: Clase {predicted_label_idx}"
    fig.suptitle(title, fontsize=16)

    for i in range(config['num_leads']):
        ax = axes[i]; signal_lead = sample_np[:, i]; saliency_lead = saliency_np[:, i]
        ax.plot(signal_lead, color='dodgerblue', linewidth=1.5)
        time_steps = np.arange(len(signal_lead))
        ax.fill_between(time_steps, np.min(signal_lead), np.max(signal_lead), where=saliency_lead > 0.5, color='tomato', alpha=0.5)
        ax.set_ylabel(lead_names[i]); ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel(f"Pasos de Tiempo (a {config['sampling_hz']} Hz)")
    fig.legend(['Señal Original', 'Zona Importante'], loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    output_dir = os.path.join('visualizations', config['run_name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"explicacion_{os.path.basename(config['ecg_path'])}.png")
    plt.savefig(output_filename)
    plt.close() # Cierra la figura para no consumir memoria
    print(f"-> Gráfico de explicación guardado en '{output_filename}'")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Genera una explicación visual para un ECG.')

    parser.add_argument('--run_name', type=str, default='resultados',
                    help='Nombre de la corrida para los resultados.')
    parser.add_argument('--s4_model_path', type=str, default='../../kardia_s4d/s4_results/S4D/model.pt',
                        help='Ruta al modelo S4 entrenado.')
    parser.add_argument('--explainer_path', type=str, default='models/my_s4d_explainer.pt',
                        help='Ruta al explicador ya entrenado.')
    parser.add_argument('--ecg_path', type=str, required=True,
                        help='Ruta al archivo ECG (.dat) para analizar.')
    parser.add_argument('--num_leads', type=int, default=12,
                        help='Número de derivaciones del ECG.')
    parser.add_argument('--sampling_hz', type=int, default=100,
                        help='Frecuencia de muestreo del ECG.')
    parser.add_argument('--duration_s', type=int, default=10,
                        help='Duración del ECG en segundos.')

    args = parser.parse_args()

    # Configuración para una corrida manual
    config = {
        'run_name': args.run_name,
        's4_model_path': args.s4_model_path,
        'explainer_path': args.explainer_path, # Ruta a un explicador ya entrenado
        'ecg_path': args.ecg_path,
        'num_leads': args.num_leads,
        'sampling_hz': args.sampling_hz,
        'duration_s': args.duration_s
    }

    sys.path.append(config['s4_model_path'].split('/s4_results')[0])
    #sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), args.s4_model_path.split('/s4_results')[0])))
    
    generate_visual_explanation(config)
