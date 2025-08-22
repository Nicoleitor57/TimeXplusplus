import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# -----------------------------------------------------------------------------
# -- CONFIGURACIÓN INICIAL --
# -----------------------------------------------------------------------------

sys.path.append('../../') 
from txai.models.bc_model import TimeXModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# -----------------------------------------------------------------------------
# -- PASO 1: CARGAR MODELO Y DATOS --
# -----------------------------------------------------------------------------

split_to_visualize = 2
model_path = os.path.join('models', f'our_bc_full_split={split_to_visualize}.pt')
data_path = os.path.join('../../datasets/FreqShape', f'split={split_to_visualize}.pt')

print(f"Cargando modelo desde: {model_path}")
print(f"Cargando datos desde: {data_path}")

sdict, config = torch.load(model_path)
D = torch.load(data_path)
test_data, _, _ = D['test']

# -----------------------------------------------------------------------------
# -- PASO 2: RECONSTRUIR EL MODELO --
# -----------------------------------------------------------------------------

mu, std = config['masktoken_stats']
config['masktoken_stats'] = (mu.to(device), std.to(device))

model = TimeXModel(
    d_inp=config['d_inp'],
    max_len=config['max_len'],
    n_classes=config['n_classes'],
    n_prototypes=config['n_prototypes'],
    gsat_r=config['gsat_r'],
    transformer_args=config['transformer_args'],
    ablation_parameters=config['ablation_parameters'],
    tau=config['tau'],
    masktoken_stats=config['masktoken_stats']
)

model.load_state_dict(sdict)
model.to(device)
model.eval()

# -----------------------------------------------------------------------------
# -- PASO 3: GENERAR LA EXPLICACIÓN PARA UNA MUESTRA --
# -----------------------------------------------------------------------------

sample_idx = 4 
sample_to_explain = test_data[sample_idx].unsqueeze(0).to(device).float()
seq_len = sample_to_explain.shape[1] 
times_to_explain = torch.arange(seq_len, device=device).unsqueeze(0).float()

with torch.no_grad():
    model_output = model(sample_to_explain, times_to_explain)
    x_select = model_output['pred']

# --- Lógica de Predicción Corregida ---
# Como el modelo predice para cada paso de tiempo, tomamos la moda como la etiqueta general
step_wise_predictions = torch.argmax(x_select, dim=-1)
predicted_label = torch.mode(step_wise_predictions).values.item()

saliency_map = model_output['ste_mask']

sample_np = sample_to_explain.cpu().numpy().squeeze()
saliency_np = saliency_map.cpu().numpy().squeeze()

# -----------------------------------------------------------------------------
# -- PASO 4: GRAFICAR Y GUARDAR EL RESULTADO --
# -----------------------------------------------------------------------------

print(f"Visualizando muestra #{sample_idx}. Predicción General del Modelo (Moda): {predicted_label}")

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(sample_np, color='dodgerblue', linewidth=2, label='Señal Original')

time_steps = np.arange(len(sample_np))
ax.fill_between(time_steps, np.min(sample_np), np.max(sample_np), where=saliency_np > 0.05,
                color='tomato', alpha=0.5,
                label='Zona Importante (Explicación de TimeX++)')

ax.set_title(f"Explicación para Muestra #{sample_idx} - Predicción General: {predicted_label}")
ax.set_xlabel("Pasos de Tiempo")
ax.set_ylabel("Valor de la Señal")
ax.legend()
plt.tight_layout()

output_filename = f"explicacion_split{split_to_visualize}_sample{sample_idx}.png"
plt.savefig(output_filename)

print(f"¡ÉXITO! Gráfico guardado como '{output_filename}'!")