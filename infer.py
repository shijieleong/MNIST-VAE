import utils
from model import VariationalAutoEncoder
from data_loader import load_data


def main():
    # Config
    INPUT_DIM = 784
    LATENT_DIM = 20
    OUTPUT_DIM = 784
    BATCHSIZE=10

    # Load Data
    _, test_loader = load_data(BATCHSIZE)

    # Initialize vae and load checkpoint
    vae = VariationalAutoEncoder(INPUT_DIM, LATENT_DIM, OUTPUT_DIM)
    vae = utils.load_checkpoint(vae, './models/trained_model.pth')
    vae.eval()
    
    # Start infer
    for image, _ in test_loader:
        output, mu, logvar = vae.forward(image)

        # Show original and output image
        utils.show_images(image, output)


if __name__ == "__main__":
    main()  