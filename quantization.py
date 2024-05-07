import numpy as np
import NumPyTorch as torch

if __name__ == "__main__":
    state_dict = torch.load_pytorch(f'./models/unet.pth')
    torch.save(state_dict, './models/unet.pkl')

    quantized_state_dict = {k: v.astype(np.float16) if isinstance(v, np.ndarray) else v for k, v in state_dict.items()}
    torch.save(quantized_state_dict, './models/unet_fp16.pkl')
