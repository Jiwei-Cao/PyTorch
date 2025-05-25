import os 
import argparse
import torch
from torchvision import transforms

import data_setup, engine, model_builder, utils


def main():
    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")

    parser.add_argument("--num_epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--hidden_units", default=10, type=int, help="hidden layer units")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--train_dir", default="data/pizza_steak_sushi/train", type=str, help="training data path")
    parser.add_argument("--test_dir", default="data/pizza_steak_sushi/test", type=str, help="testing data path")

    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate
    train_dir = args.train_dir
    test_dir = args.test_dir

    print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")
    print(f"[INFO] Training data file: {train_dir}")
    print(f"[INFO] Testing data file: {test_dir}")

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Dataloaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # Model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="05_going_modular_script_mode_tinyvgg_model.pth")


if __name__ == "__main__":
    main()
