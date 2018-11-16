import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.densenet import Atlas_DenseNet
from models.resnet import ResNet

from utils.dataloader import get_test_loader

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--imsize', default=256, type=int, 
                    help='image size')
parser.add_argument('--batch_size', default=200, type=int, 
                    help='size of batches')
parser.add_argument('--model_name', default='resnet', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='run5_FocalLoss', type=str,
                    help='name of experiment for saving files')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = './model_weights/best_{}_{}.pth'.format(args.model_name,
                                                       args.exp_name)
OUT_FILE = './subm/' + os.path.basename(model_path.replace('pth', 'csv'))
test_submission_path = f"./data/sample_submission.csv"
print('Saving to {}'.format(OUT_FILE))

def test():
    test_loader = get_test_loader(imsize=args.imsize, batch_size=args.batch_size)
    # net = Atlas_DenseNet(model = args.model_name, bn_size=4)
    net = ResNet()
    net = nn.DataParallel(net).to(device)
    cudnn.benchmark = True

    net.load_state_dict(torch.load(model_path, 
                                   map_location=lambda storage, 
                                    loc: storage))

    net.eval()

    a = None
    out = []
    with torch.no_grad():
        for data in test_loader:
            # move to GPU if available
            test_imgs = data[0].to(device)

            # compute model output
            output_batch_raw = net(test_imgs)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch_raw = output_batch_raw.data.cpu().numpy()
            output_batch = (output_batch_raw > 0.0).astype(np.int32)
            for i in range(output_batch.shape[0]):
                output_batch_str = ' '.join(str(v) for v in np.nonzero(output_batch[i])[0].tolist())
                if output_batch_str == '':
                    output_batch_str = str(output_batch_raw[i, :].argmax())
                out.append(output_batch_str)

    test_df = pd.read_csv(test_submission_path)
    test_df.Predicted = out
    test_df.to_csv(OUT_FILE, index=False)

if __name__ == '__main__':
    test()
    

    