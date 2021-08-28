import json
import numpy as np
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    # Modified: Added this class from this stack overflow suggestions to fix the json dump issue
    #  'np.float32 is not JSON serializable'
    #   https://stackoverflow.com/a/49677241
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)


def plot_metrics(model_history, stop=50):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    axes.plot(range(stop), model_history['loss'], label='Training', color='#FF533D')
    axes.plot(range(stop), model_history['val_loss'], label='Validation', color='#03507E')
    axes.set_title('Loss')
    axes.set_ylabel('Loss')
    axes.set_xlabel('Epoch')
    axes.legend(loc='upper right')
    fig.savefig('train_val_losses.png', dpi=fig.dpi)
