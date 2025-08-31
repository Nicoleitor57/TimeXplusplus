

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os

# ===================================================================
# >> CONFIGURACIÓN DE RUTAS (COMPLETA ESTAS LÍNEAS) <<
# ===================================================================
PATH_A_TU_PROYECTO_S4D_ECG = '../../S4D-ECG' 
PATH_A_TU_MODELO_S4D_ENTRENADO = '../../S4D-ECG/s4_results/S4D/model.pt'
PATH_A_TUS_EMBEDDINGS = '../../S4D-ECG/s4_results/embeddings/ecg_embeddings_testset.pt'
# ===================================================================

# Añadimos las rutas necesarias para importar tus clases y las de TimeX++
sys.path.append(PATH_A_TU_PROYECTO_S4D_ECG)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Importamos las piezas que necesitamos
from src.models.s4.s4d import S4D 
from Train import dropout_fn # Asumiendo que dropout_fn está en tu Train.py
from txai.models.mask_generators.maskgen import MaskGenerator
from txai.utils.predictors.loss import Poly1CrossEntropyLoss
from txai.utils.predictors.loss_cl import LabelConsistencyLoss, EmbedConsistencyLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# >> TU CLASE S4Model (COPIADA AQUÍ PARA SER AUTOCONTENIDO) <<
# -----------------------------------------------------------------------------
class S4Model(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=128, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, times, **kwargs):
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        embedding_secuencia = x 
        pooled_output = x.mean(dim=1)
        final_prediction = self.decoder(pooled_output)
        final_prediction = nn.functional.sigmoid(final_prediction)
        return final_prediction, embedding_secuencia, pooled_output

# -----------------------------------------------------------------------------
# >> NUESTRO NUEVO MODELO EXPLICADOR A MEDIDA <<
# -----------------------------------------------------------------------------
class S4D_Explainer(nn.Module):
    def __init__(self, s4_model, d_model, max_len, d_inp):
        super().__init__()
        # 1. Contiene tu modelo S4D como la "caja negra"
        self.s4_model = s4_model
        
        # 2. Contiene el generador de máscaras de TimeX++
        # ¡Lo inicializamos con el d_model correcto (128)!
        self.mask_generator = MaskGenerator(d_z=d_model, max_len=max_len, d_inp=d_inp)

    def forward(self, x, times):
        # Congelamos el S4D para que no se entrene
        self.s4_model.eval()
        with torch.no_grad():
            #prediction, embedding = self.s4_model(x, times)
            prediction, embedding, pooled_embedding = self.s4_model(x, times)

        # --- ¡CORRECCIÓN FINAL! ---
        # El MaskGenerator espera los tensores con una forma específica (T, B, d)
        # La entrada principal ('memory') debe ser el embedding de alta dimensión.
        # La entrada objetivo ('tgt') es la que usa para la decodificación.
        # La forma más robusta es pasarle el propio embedding como 'tgt' y 'memory'.
        
        # Transponemos para que tengan la forma (Tiempo, Batch, Dimensión)
        x_T = x.transpose(0, 1)
        times_T = times.transpose(0, 1)
        embedding_T = embedding.transpose(0, 1)

        # Llamamos al generador de máscaras con los argumentos correctos.
        # 'z_seq' (memory) es el embedding de S4D.
        # 'src' (tgt) también debe ser un embedding. Usaremos el mismo.
        mask_tuple = self.mask_generator(z_seq=embedding_T, src=embedding_T, times=times_T)

        
        
        # El MaskGenerator devuelve una tupla, la máscara real es el segundo elemento.
        mask = mask_tuple[1]
        
        # Devolvemos todo lo que necesitamos en un diccionario claro
        #return {'prediction': prediction, 'mask': mask}
        return {'prediction': prediction, 'mask': mask, 'pooled_embedding': pooled_embedding}
        


class Poly1BCELoss(nn.Module):
    """
    Implementación de Poly1 Loss para problemas de clasificación multi-etiqueta.
    Combina la fórmula de Poly1 con Binary Cross-Entropy.
    """
    def __init__(self, epsilon=2.0, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, labels):
        # Usamos BCEWithLogitsLoss por estabilidad numérica.
        # 'logits' son las salidas de tu modelo ANTES de la función sigmoide.
        # Tu modelo S4D ya aplica sigmoide, así que lo revertimos con torch.logit
        # para mayor precisión.
        if torch.all((logits >= 0) & (logits <= 1)): # Si ya se aplicó sigmoide
             logits = torch.logit(logits, eps=1e-6)

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        # p_t es la probabilidad que el modelo asigna a la clase correcta.
        # Se puede obtener a partir de la pérdida BCE: p_t = exp(-BCE)
        pt = torch.exp(-bce_loss)
        
        # Esta es la modificación de Poly1 que mejora el entrenamiento
        poly1_loss = bce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            return poly1_loss.mean()
        elif self.reduction == 'sum':
            return poly1_loss.sum()
        else:
            return poly1_loss

# -----------------------------------------------------------------------------
# >> BUCLE DE ENTRENAMIENTO SIMPLIFICADO <<
#-----------------------------------------------------------------------------
# def train_explainer(explainer, dataloader, optimizer, clf_criterion, device):
#     explainer.train()
#     total_loss = 0
#     pbar = tqdm(dataloader, desc="Entrenando Explicador")

#     for x, times, y in pbar:
#         x, times, y = x.to(device), times.to(device), y.to(device)
        
#         optimizer.zero_grad()
        
#         # Obtenemos la salida de nuestro explicador a medida
#         output = explainer(x, times)
#         mask = output['mask']
        
#         # Aquí es donde se aplica el principio de TimeX++
#         # 1. Pérdida de clasificación: La predicción del modelo NO debe cambiar
#         #    cuando se enmascara la señal (esta es una simplificación, la real
#         #    usa consistencia, pero empecemos por aquí).
        
#         # 2. Pérdida de dispersión (Sparsity Loss): ¡Queremos que la máscara sea pequeña!
#         #    Esta es la clave para obtener explicaciones limpias.
#         sparsity_loss = torch.mean(torch.abs(mask))
        
#         # Combinamos las pérdidas (aquí puedes ajustar los pesos)
#         # Por ahora, nos centraremos solo en la dispersión
#         loss = sparsity_loss # Simplificado por ahora
        
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         pbar.set_postfix({'loss': loss.item(), 'sparsity': sparsity_loss.item()})
        
#     return total_loss / len(dataloader)





# -----------------------------------------------------------------------------
# >> BUCLE DE ENTRENAMIENTO FINAL (CON PÉRDIDA COMPLETA DE TIMEX++) <<
# # -----------------------------------------------------------------------------
def train_explainer(explainer, dataloader, optimizer, clf_criterion, device):
    explainer.train()
    total_loss, total_fidelity, total_sparsity, total_consistency = 0, 0, 0, 0
    pbar = tqdm(dataloader, desc="Entrenando Explicador (Pérdida Completa)")

    # Hiperparámetros para ponderar las pérdidas (puedes ajustarlos)
    beta_fidelity = 1.0 # Peso para la pérdida de fidelidad
    beta_sparsity = 0.5 # Peso para la pérdida de dispersión
    beta_consistency = 1.0 # Peso para la pérdida de consistencia

    # Criterios de pérdida para la consistencia
    consistency_criterion_label = LabelConsistencyLoss()
    consistency_criterion_embed = EmbedConsistencyLoss()

    for x, times, y in pbar:
        x, times, y = x.to(device), times.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # --- Paso 1: Forward Pass Original ---
        output_original = explainer(x, times)
        prediction_original = output_original['prediction']
        mask_original = output_original['mask']
        #########  EMBEDDING ORIGINAL  #########
        embedding_original = output_original['pooled_embedding']
        #########  FIN EMBEDDING ORIGINAL  #########

        # --- Paso 2: Creamos una versión "aumentada" de la entrada ---
        # Añadimos un poco de ruido para crear una señal muy similar
        x_augmented = x + (torch.randn_like(x) * 0.05)
        
        # --- Paso 3: Forward Pass de la Versión Aumentada ---
        output_augmented = explainer(x_augmented, times)
        prediction_augmented = output_augmented['prediction']
        mask_augmented = output_augmented['mask']
        embedding_augmented = output_augmented['pooled_embedding']

        # --- Cálculo de las 3 Pérdidas ---
        
        # 1. Pérdida de Dispersión (Sparsity Loss) - Queremos que la máscara sea pequeña
        sparsity_loss = torch.mean(torch.abs(mask_original))
        
        # 2. Pérdida de Fidelidad (Fidelity Loss) - La predicción no debe cambiar al enmascarar
        signal_masked = x * mask_original
        prediction_after_masking, _, _ = explainer.s4_model(signal_masked, times)
        #fidelity_loss = clf_criterion(prediction_after_masking, prediction_original)
        fidelity_loss = clf_criterion(prediction_after_masking, y)
        

        # 3. Pérdida de Consistencia (Consistency Loss) - Explicaciones similares para entradas similares
        # Consistencia a nivel de etiqueta
        consistency_loss_label = consistency_criterion_label(prediction_original, prediction_augmented)


        ############
        #mask_original_agg = mask_original.mean(dim=1)
        #mask_augmented_agg = mask_augmented.mean(dim=1)
        ############

        # Consistencia a nivel de máscara (embedding)
        #consistency_loss_embed = consistency_criterion_embed(mask_augmented_agg, mask_original_agg)
        consistency_loss_embed = consistency_criterion_embed(embedding_original, embedding_augmented)
        
        total_consistency_loss = consistency_loss_label + consistency_loss_embed
        
        # --- Combinamos todas las pérdidas ---mask_augmented_agg
        loss = (beta_fidelity * fidelity_loss) + \
               (beta_sparsity * sparsity_loss) + \
               (beta_consistency * total_consistency_loss)
        
        loss.backward()
        optimizer.step()
        
        # Guardamos valores para el log
        total_loss += loss.item()
        total_fidelity += fidelity_loss.item()
        total_sparsity += sparsity_loss.item()
        total_consistency += total_consistency_loss.item()
        
        pbar.set_postfix({
            'loss': loss.item(), 
            'fidelidad': fidelity_loss.item(), 
            'dispersión': sparsity_loss.item(),
            'consistencia': total_consistency_loss.item()
        })
        
    # Imprimimos el promedio de las pérdidas de la época
    num_batches = len(dataloader)
    print(f"Época completada. Promedios -> Loss: {total_loss / num_batches:.4f}, Fidelidad: {total_fidelity / num_batches:.4f}, Dispersión: {total_sparsity / num_batches:.4f}, Consistencia: {total_consistency / num_batches:.4f}")
        
    return total_loss / len(dataloader)



# -----------------------------------------------------------------------------
# >> EJECUCIÓN PRINCIPAL <<
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # --- Carga de Datos ---
    print("==> Cargando datos de ECG...")
    data_dict = torch.load(PATH_A_TUS_EMBEDDINGS)
    original_signals = data_dict['original_signals']
    labels = data_dict['labels']
    d_inp, max_len, n_classes = original_signals.shape[2], original_signals.shape[1], labels.shape[1]
    
    times = torch.arange(max_len).unsqueeze(0).repeat(original_signals.shape[0], 1)
    
    dataset = TensorDataset(original_signals, times, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- Construcción de Modelos ---
    print(f"==> Cargando tu modelo S4D-ECG...")
    s4_model_trained = S4Model(d_input=d_inp, d_output=n_classes, d_model=128, n_layers=4)
    s4_model_trained.load_state_dict(torch.load(PATH_A_TU_MODELO_S4D_ENTRENADO, map_location=device))
    s4_model_trained.to(device)
    
    print("==> Construyendo el nuevo modelo explicador...")
    explainer = S4D_Explainer(s4_model=s4_model_trained, d_model=128, max_len=max_len, d_inp=d_inp)
    explainer.to(device)
    
    # --- Entrenamiento del Explicador ---
    # Solo entrenamos los parámetros del MaskGenerator
    optimizer = torch.optim.AdamW(explainer.mask_generator.parameters(), lr=1e-3)
    #clf_criterion = Poly1CrossEntropyLoss(num_classes=n_classes)
    #clf_criterion = nn.BCELoss()
    clf_criterion = Poly1BCELoss(epsilon=2.0)
    
    epochs = 30 # Empecemos con pocas épocas
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        avg_loss = train_explainer(explainer, dataloader, optimizer, clf_criterion, device)
        print(f"Epoch {epoch+1} completada. Pérdida promedio: {avg_loss:.4f}")
    
    # --- Guardado del Explicador Entrenado ---
    save_path = 'models/my_s4d_explainer.pt'
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(explainer.state_dict(), save_path)
    print(f"\n¡ÉXITO! Explicador entrenado y guardado en '{save_path}'")