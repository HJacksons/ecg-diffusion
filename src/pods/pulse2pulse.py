from networks.Pulse2Pulse import WaveGANGenerator, WaveGANDiscriminator
from contracts.pod import PodContract
from torch.autograd import Variable
import configuration as conf
import helpers as helpers
from torch import autograd
import torch


class Pulse2PulsePod(PodContract):
    def __init__(self, lr):
        self.model = WaveGANGenerator().to(device=conf.DEVICE)
        self.netD = WaveGANDiscriminator().to(device=conf.DEVICE)

        self.optimizerG = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.9))
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.9))

        self.train_G_flag = False

    def batch_processing(self, batch, leadsI_VIII, feature):
        if (batch+1) % 5 == 0:
            self.train_G_flag = True

        # Set Discriminator parameters to require gradients.
        for p in self.netD.parameters():
            p.requires_grad = True

        one = torch.tensor(1, dtype=torch.float)
        neg_one = one * -1

        one = one.to(device=conf.DEVICE)
        neg_one = neg_one.to(device=conf.DEVICE)

        #############################
        # (1) Train Discriminator
        #############################
        
        real_ecgs = leadsI_VIII.to(device=conf.DEVICE)
        b_size = real_ecgs.size(0)

        self.netD.zero_grad()

        # Noise
        noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
        noise = noise.to(device=conf.DEVICE)
        noise_Var = Variable(noise, requires_grad=False)

        # a) compute loss contribution from real training data
        D_real = self.netD(real_ecgs)
        D_real = D_real.mean()  # avg loss
        D_real.backward(neg_one)  # loss * -1

        # b) compute loss contribution from generated data, then backprop.
        fake = autograd.Variable(self.model(noise_Var).data)
        D_fake = self.netD(fake)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # c) compute gradient penalty and backprop
        gradient_penalty = helpers.calc_gradient_penalty(self.netD, real_ecgs,
                                                fake.data, b_size, 10.0,
                                                use_cuda=True)
        gradient_penalty.backward(one)

        # Compute cost * Wassertein loss..
        D_cost_train = D_fake - D_real + gradient_penalty
        D_wass_train = D_real - D_fake

        # Update gradient of discriminator.
        self.optimizerD.step()

        #############################
        # (3) Train Generator
        #############################
        if self.train_G_flag:
            # Prevent discriminator update.
            for p in self.netD.parameters():
                p.requires_grad = False

            # Reset generator gradients
            self.model.zero_grad()

            # Noise
            noise = torch.Tensor(b_size, 8, 5000).uniform_(-1, 1)
            
            noise = noise.to(device=conf.DEVICE)
            noise_Var = Variable(noise, requires_grad=False)

            fake = self.model(noise_Var)
            G = self.netD(fake)
            G = G.mean()

            # Update gradients.
            G.backward(neg_one)
            G_cost = -G

            self.optimizerG.step()

            # Record costs
            G_cost_cpu = G_cost.data
            train_G_flag = False
                
            return G_cost_cpu.cpu()
        return None

    def sampling(self, load_pretrained_model=False):
        trained_model_path = f"{conf.MODELS_FOLDER}/Pulse2pulse_epoch530.pt"

        if load_pretrained_model:
            self.model.load_state_dict(
                torch.load(
                    trained_model_path,
                    map_location=torch.device(conf.DEVICE)
                )
            )

        self.model.eval()
        with torch.inference_mode():
            noise = torch.Tensor(1, 8, 5000).uniform_(-1, 1).to(device=conf.DEVICE)
            fake = self.model(noise)

        return fake

    def validation(self):
        return None
