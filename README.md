# Project Ukiyo-e: Generative Art with NCA and StyleGAN

This project aims to create a generative art system by combining a Neural Cellular Automata (NCA) and a StyleGAN. The two models will be trained on a dataset of Ukiyo-e images, and they will be interconnected to learn from and compete with each other. The training process and the generated outputs will be visualized through a web-based UI.

## Core Components

- **Neural Cellular Automata (NCA):** A model that learns to "grow" images from a single seed, following local rules.
- **StyleGAN:** A powerful generative adversarial network known for producing high-quality and controllable images.
- **Latent Space Coupling:** A mechanism to link the latent spaces of the NCA and StyleGAN, allowing them to influence each other.
- **Competitive Training:** A training regimen where the models are pitted against each other to drive learning.
- **Visualization UI:** A web interface to monitor the training process and interact with the models.

## Dataset

The project uses a dataset of Ukiyo-e art, located in the `data/ukiyo-e` directory. 