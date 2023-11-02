import numpy as np


def calculate_mape(output, target, y_scaler):
    output_descaled = y_scaler.inverse_transform(output.cpu().detach().numpy())
    target_descaled = y_scaler.inverse_transform(target.cpu().detach().numpy())

    output = np.expm1(output_descaled)
    target = np.expm1(target_descaled)

    mape = np.mean(np.abs((target - output) / target))
    return mape
