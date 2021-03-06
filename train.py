import sys
import pyro
import argparse
import pandas as pd
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from VAE import FormatVAE
from utilities.name_dataset import NameDataset
from utilities.plot import plot_losses

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=2048)
parser.add_argument('--num_epochs', help='number of epochs',
                    type=int, default=1000)
parser.add_argument('--learning_rate', help='learning rate',
                    type=float, default=1.e-20)
parser.add_argument('--max_input_size',
                    help='max string length', type=int, default=18)
args = parser.parse_args()

pyro.enable_validation(True)
NUM_EPOCHS = args.num_epochs
LR = args.learning_rate
BATCH_SIZE = args.batch_size
MAX_INPUT_STRING_LEN = args.max_input_size
ADAM_CONFIG = {'lr': LR}


def weights_for_balanced_class(df, target_column):
    """
    Assign higher weights to rows whose class is not prevalent
    to sample each class equally with DataLoader
    """
    target = df
    num_classes = target.nunique()
    counts = [0] * num_classes
    for row_class in target:
        counts[row_class] += 1
    class_weights = [0] * num_classes
    for i, count in enumerate(counts):
        class_weights[i] = len(target)/count
    weights = [0] * len(target)
    for i, row_class in enumerate(target):
        weights[i] = class_weights[row_class]
    return weights


def simple_elbo_kl_annealing(model, guide, *args, **kwargs):
    # get the annealing factor and latents to anneal from the keyword
    # arguments passed to the model and guide
    annealing_factor = kwargs.pop('annealing_factor', 1.0)
    latents_to_anneal = kwargs.pop('latents_to_anneal', [])
    # run the guide and replay the model against the guide
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    elbo = 0.0
    # loop through all the sample sites in the model and guide trace and
    # construct the loss; note that we scale all the log probabilities of
    # samples sites in `latents_to_anneal` by the factor `annealing_factor`
    for site in model_trace.nodes.values():
        if site["type"] == "sample":
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            elbo = elbo + factor * site["fn"].log_prob(site["value"]).sum()
    for site in guide_trace.nodes.values():
        if site["type"] == "sample":
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            elbo = elbo - factor * site["fn"].log_prob(site["value"]).sum()
    return -elbo


def train_one_epoch(loss, dataloader, epoch_num):
    total_loss = 0.
    i = 1
    for batch in dataloader:
        batch_loss = loss.step(batch)/len(batch)
        # batch_loss = loss.step(batch, annealing_factor=0.2,
        #                        latents_to_anneal=["z"])/len(batch)
        total_loss += batch_loss
        if i % 10 == 0:
            print(
                f"Epoch {epoch_num} {i}/{len(dataloader)} Loss: {batch_loss}")
        i += 1

    avg_loss = total_loss/len(dataloader)
    return avg_loss


dataset = NameDataset("data/FirstNames.csv", "name",
                      max_string_len=MAX_INPUT_STRING_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vae = FormatVAE(encoder_hidden_size=512,
                decoder_hidden_size=512, mlp_hidden_size=128)

if len(sys.argv) > 1:
    vae.load_checkpoint(filename=sys.argv[1].split('/')[-1])

svi_loss = SVI(vae.model, vae.guide, Adam(ADAM_CONFIG), loss=Trace_ELBO())
# svi_loss = SVI(vae.model, vae.guide, Adam(ADAM_CONFIG), loss=simple_elbo_kl_annealing)


epoch_losses = []
for e in range(NUM_EPOCHS):
    print("===========================")
    print(f"Epoch {e} Generated Names")
    print("===========================")
    for _ in range(5):
        print(f"- {vae.model(None)[0]}")
    avg_loss = train_one_epoch(svi_loss, dataloader, e)
    vae.save_checkpoint(filename="test.pth.tar")
    epoch_losses.append(avg_loss)
    plot_losses(epoch_losses, filename="test.png")
