from tensorflow.keras.utils import plot_model
import tensorflow.keras.models as models

model = models.load_model('../Modelle/model_0.781.keras')

plot_model(model,
           to_file='model_architecture.png',
           show_shapes=True,  # zeigt Ein-/Ausgabe-Shapes
           show_layer_names=True,  # zeigt Layer-Namen
           rankdir='TB',      # 'TB' (top-bottom) oder 'LR' (left-right)
           dpi=300)          # höhere Auflösung