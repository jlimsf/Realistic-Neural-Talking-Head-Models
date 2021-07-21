
"""Main"""

import wandb
wandb.login()
from sklearn.manifold import TSNE

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.ion()
import os

from dataset.dataset_class import PreprocessDataset, UBCFashion
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *
from tqdm import tqdm

from params.params import K, path_to_chkpt, path_to_backup, path_to_Wi, batch_size, path_to_preprocess, frame_shape

def view_tsne(embedding_layer):
    tsne = TSNE(perplexity=15, n_components=2, n_iter=3500, random_state=32)
    embeddings_en_2d = tsne.fit_transform(embedding_layer.detach().cpu().numpy())

    return embeddings_en_2d

os.environ['WANDB_CONFIG_DIR'] = '/home/ubuntu/playpen/'
wandb.init(project='UBC_Fashion_fsgan')


"""Create dataset and net"""
display_training = False
device = torch.device("cuda:0")
cpu = torch.device("cpu")

dataset = UBCFashion(K=8,
                pose_root='/home/ubuntu/playpen/pose_playground/pose_white/train/',
                rgb_root='../articulated-animation/data/fashion_png/train',
                transforms=transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

# (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print ("Working with {} videos".format(len(dataset)))

dataLoader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        drop_last = True)


#
# dataset = PreprocessDataset(K=K, path_to_preprocess=path_to_preprocess, path_to_Wi=path_to_Wi)
# dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
#                         num_workers=8,
#                         pin_memory=True,
#                         drop_last = True)

G = nn.DataParallel(Generator(frame_shape).to(device))
E = nn.DataParallel(Embedder(frame_shape).to(device))
D = nn.DataParallel(Discriminator(dataset.__len__(), path_to_Wi).to(device))

# G = Generator(frame_shape).to(device)
# E = Embedder(frame_shape).to(device)
# D = Discriminator(dataset.__len__(), path_to_Wi).to(device)



G.train()
E.train()
D.train()

optimizerG = optim.Adam(params = list(E.parameters()) + list(G.parameters()),
                        lr=0.0001, betas=(0.5, 0.999),
                        amsgrad=False)
optimizerD = optim.Adam(params = D.parameters(),
                        lr=0.0001, betas=(0.5, 0.999),
                        amsgrad=False)

print (optimizerG)
print (optimizerD)
"""Criterion"""
criterionG = LossG(VGGFace_body_path='content/vggface/Pytorch_VGGFACE_IR.py',
                   VGGFace_weight_path='content/vggface/Pytorch_VGGFACE.pth', device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
lossesD = []
i_batch_current = 0

num_epochs = 10000

wandb.watch(G, log_freq=1)
wandb.watch(E, log_freq=1)
wandb.watch(D, log_freq=1)



#initiate checkpoint if inexistant

if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    G.apply(init_weights)
    D.apply(init_weights)
    E.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'lossesD': lossesD,
            'E_state_dict': E.module.state_dict(),
            'G_state_dict': G.module.state_dict(),
            'D_state_dict': D.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
            }, path_to_chkpt)
    print('...Done')


# """Loading from past checkpoint"""
# checkpoint = torch.load(path_to_chkpt, map_location=cpu)
# E.module.load_state_dict(checkpoint['E_state_dict'])
# G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
# D.module.load_state_dict(checkpoint['D_state_dict'])
# epochCurrent = checkpoint['epoch']
# lossesG = checkpoint['lossesG']
# lossesD = checkpoint['lossesD']
# num_vid = checkpoint['num_vid']
# i_batch_current = checkpoint['i_batch'] +1
# optimizerG.load_state_dict(checkpoint['optimizerG'])
# optimizerD.load_state_dict(checkpoint['optimizerD'])

G.train()
E.train()
D.train()

"""Training"""
batch_start = datetime.now()
pbar = tqdm(dataLoader, leave=True, initial=0)
if not display_training:
    matplotlib.use('agg')


for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)
    pbar.set_postfix(epoch=epoch)

    running_match_loss = 0
    running_cnt_loss = 0
    running_adv_loss = 0
    running_g_loss = 0
    running_d_loss = 0
    running_l1_loss = 0

    for i_batch, (f_lm, x, g_y, i, poses, rgbs) in enumerate(pbar, start=0):

        f_lm = f_lm.to(device)
        x = x.to(device)
        g_y = g_y.to(device)
        i = i.to(device)

        if i_batch % 1 == 0:
            # with torch.autograd.enable_grad():
                #zero the parameter gradients
            optimizerG.zero_grad()


            #forward
            # Calculate average encoding vector for video
            f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224
            e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxK,512,1
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,K,512,1
            e_hat = e_vectors.mean(dim=1)

            #train G and D
            x_hat = G(g_y, e_hat)

            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            r, D_res_list = D(x, g_y, i)

            lossG, this_loss_match, this_loss_adv, this_loss_cnt, this_l1_loss = \
                                    criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D.module.W, i)

            lossG.backward(retain_graph=False)
            optimizerG.step()

            optimizerD.zero_grad()
            r_hat, D_hat_res_list = D(x_hat.detach(), g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D(x, g_y, i)
            lossDreal = criterionDreal(r)

            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            optimizerD.step()


            running_match_loss += this_loss_match
            running_cnt_loss += this_loss_cnt
            running_adv_loss += this_loss_adv
            running_d_loss += lossD.item()
            running_g_loss += lossG.item()
            running_l1_loss += this_l1_loss


            # optimizerD.zero_grad()
            # r_hat, D_hat_res_list = D(x_hat, g_y, i)
            # lossDfake = criterionDfake(r_hat)
            #
            # r, D_res_list = D(x, g_y, i)
            # lossDreal = criterionDreal(r)
            #
            # lossD = lossDfake + lossDreal
            # lossD.backward(retain_graph=False)
            # optimizerD.step()





        # Output training stats
        if i_batch % 1 == 0 and i_batch > 0:

            pbar.set_postfix(epoch=epoch, r=r.mean().item(), rhat=r_hat.mean().item(), lossG=lossG.item())



        if i_batch % 1000 == 999:
            lossesD.append(lossD.item())
            lossesG.append(lossG.item())



            print('Saving latest...')
            torch.save({
                    'epoch': epoch,
                    'lossesG': lossesG,
                    'lossesD': lossesD,
                    'E_state_dict': E.module.state_dict(),
                    'G_state_dict': G.module.state_dict(),
                    'D_state_dict': D.module.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': i_batch,
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()
                    }, path_to_chkpt)


    if epoch%1 == 0:
        print('Saving latest...')
        torch.save({
                'epoch': epoch+1,
                'lossesG': lossesG,
                'lossesD': lossesD,
                'E_state_dict': E.module.state_dict(),
                'G_state_dict': G.module.state_dict(),
                'D_state_dict': D.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': i_batch,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
                }, path_to_backup)

        print('...Done saving latest')

        epoch_match_loss = running_match_loss/len(dataLoader)
        epoch_cnt_loss = running_cnt_loss/len(dataLoader)
        epoch_adv_loss = running_adv_loss/len(dataLoader)
        epoch_g_loss = running_g_loss/len(dataLoader)
        epoch_d_loss = running_d_loss/len(dataLoader)
        epoch_l1_loss = running_l1_loss/len(dataLoader)

        tsne_embeddings = view_tsne(D.module.W)
        data = [[x, y] for (x, y) in zip(tsne_embeddings[:,0], tsne_embeddings[:,1])]
        table = wandb.Table(data=data, columns = ["x", "y"])
        scatter = wandb.plot.scatter(table, "class_x", "class_y")

        wandb.log({
                    'generator loss': epoch_g_loss,
                    'discrim loss': epoch_d_loss,
                    'realism score': r.mean().item(),
                    'adversarial_loss': epoch_adv_loss,
                    'match_loss': epoch_match_loss,
                    'l1_loss': epoch_l1_loss,
                    'content_loss' : epoch_cnt_loss,
                    'GT_im': [wandb.Image(_) for _ in x],
                    'GT_pose': [wandb.Image(_) for _ in g_y],
                    'generated_images': [wandb.Image(_) for _ in x_hat],
                    'input_images': [wandb.Image(_) for _ in f_lm[:,:,0,:,:,:]],
                    'pose_images': [wandb.Image(_) for _ in f_lm[:,:,1,:,:,:]]  })
        wandb.log({'tsne embeddings': scatter})
