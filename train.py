import argparse
import os
import datetime
import torch
import numpy as np
from statistics import mean
from genomedata import GenomeData
from datasets import *
import autoencoder
from loss import tobit_loss


def main(args):
    start_t = datetime.datetime.now()
    torch.cuda.set_device(args.gpu)

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.mkdir(args.output)

    gd = GenomeData(args.input)
    gd.load_data()
    gd.preprocess_data()

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    setup_seed(args.seed)

    train_set = gd.data_rc_all.copy()

    model = autoencoder.AE(train_set.shape[1], z_dim=args.latent_dim, seg_max=args.max_seg_len).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=0,
                                 amsgrad=False)

    model.train()
    train_loss = []
    epoch_list = []
    for epoch in range(args.epochs):
        losses = []
        epoch_list.append(epoch)
        loader = Data.DataLoader(CellDataSet(train_set), args.batch_size, True)
        for x in loader:
            x = x.cuda()
            x = x.unsqueeze(1)
            z, mu, sigma = model(x)

            x = x.squeeze()
            loss_rec = tobit_loss(x, mu, sigma)
            loss = loss_rec

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = mean(losses)
        train_loss.append(np.array(mean_loss))
        print("epoch: " + str(epoch) + " loss:" + str(mean_loss))

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.mkdir(args.output)
    ll_file = output_dir + '/loss.txt'
    if os.path.isfile(ll_file):
        os.remove(ll_file)
    file_o = open(ll_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(train_loss, (1, len(train_loss)))], fmt='%f', delimiter=',')
    file_o.close()

    # get reconstructed data
    model.eval()
    train_set = gd.data_rc_all.copy()
    loader = Data.DataLoader(CellDataSet(train_set), args.batch_size, False)
    features = []
    reconstructed = []
    for x in loader:
        x = x.cuda()
        x = x.unsqueeze(1)
        with torch.no_grad():
            encodings, mu, sigma = model(x)
            encodings = encodings.cpu().detach().numpy()
            mu = mu.cpu().detach().numpy()
            if len(features) == 0:
                features = encodings
            else:
                features = np.r_[features, encodings]
            if len(reconstructed) == 0:
                reconstructed = mu
            else:
                reconstructed = np.r_[reconstructed, mu]

    # save results
    rc_file = output_dir + '/corrected_rc.txt'
    if os.path.isfile(rc_file):
        os.remove(rc_file)
    file_o = open(rc_file, 'a')
    np.savetxt(file_o, np.c_[reconstructed], fmt='%.3f', delimiter=' ')
    file_o.close()

    latent_file = output_dir + '/latent.txt'
    file_o = open(latent_file, 'w')
    np.savetxt(file_o, np.c_[features], fmt='%.3f', delimiter=',')
    file_o.close()

    barcode_file = output_dir + '/barcode.txt'
    file_o = open(barcode_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(gd.barcodes, (1, len(gd.barcodes)))], fmt='%s', delimiter=',')
    file_o.close()

    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t - start_t).seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="scTCA")
    parser.add_argument('--gpu', type=int, default=0, help='which GPU to use.')
    parser.add_argument('--epochs', type=int, default=200, help='number of epoches to train the scTCA.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--latent_dim', type=int, default=5, help='the latent dimension.')
    parser.add_argument('--max_seg_len', type=int, default=500, help='the maximum length of subsequence for stepwise self attention.')
    parser.add_argument('--seed', type=int, default=0, help='random seed.')
    parser.add_argument('--input', type=str, default='', help='input file.')
    parser.add_argument('--output', type=str, default='', help='a directory to save results.')
    args = parser.parse_args()
    main(args)
