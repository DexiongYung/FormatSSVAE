import sys
from VAE import FormatVAE

NUM_GENERATION = 5
vae = FormatVAE(encoder_hidden_size=256,
                decoder_hidden_size=64, mlp_hidden_size=32)
vae.load_checkpoint(filename='test.pth.tar')

print("================")
print(f"Generated Names")
print("================")
for _ in range(int(NUM_GENERATION)):
    print(f"- {vae.model(None)[0]}")

print('input = lee')

for i in range(10):
    print(vae.generate(['lee'])[0].replace('1', ''))
