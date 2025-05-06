import torch
import torch.optim as optim

from data_loader import load_data
from model import VariationalAutoEncoder
import utils

def train(vae, train_loader, device, beta, num_epochs, lr_rate):
    optimizer = optim.Adam(vae.parameters(), lr=lr_rate)
    
    # Start training
    for epoch in range(num_epochs):
        print(f'Training epoch {epoch}...')

        # Reset running loss
        running_loss = 0

        # Traning 1 batch of data
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # Get output
            output, mu, logvar = vae.forward(images)

            # Calculate loss
            optimizer.zero_grad()
            total_loss, recons_loss, kld_loss = vae.compute_loss(output, images, mu, logvar, epoch, 4)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss

        print(f'Loss: {running_loss/len(train_loader):.4f}, Recons Loss:{recons_loss}, KLD Loss:{kld_loss}')


def main():
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 784
    LATENT_DIM = 20
    OUTPUT_DIM = 784
    BETA = 0.5
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    LR_RATE = 0.001

    # Load data
    train_loader, test_loader = load_data(BATCH_SIZE)

    # Create VAE
    vae = VariationalAutoEncoder(INPUT_DIM, LATENT_DIM, OUTPUT_DIM)
    vae.to(DEVICE)
    train(vae, train_loader, DEVICE, BETA, NUM_EPOCHS, LR_RATE)
    #vae = utils.load_checkpoint(vae, './models/trained_model.pth')

    # Save Model
    utils.save_checkpoint(vae)
    
    return


if __name__ == "__main__":
    main()