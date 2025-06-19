# Project Backlog

This document outlines the development plan and tracks the progress of the Ukiyo-e project.

## Priority 1: Core Model Implementation (Complete)

- [x] **Data Loader:** Implement a data loader to process the Ukiyo-e images.
- [x] **NCA Model:** Create the Neural Cellular Automata model (`nca.py`).
- [x] **StyleGAN Model:** Implement the StyleGAN model (`stylegan.py`).
- [x] **Basic Training Script:** Set up a basic `train.py` to test each model individually.

## Priority 2: Direct Model Integration

- [ ] **Shared Core Module:** Create a `SharedBrain` module with weights to be used by both models.
- [ ] **Integrated Models:**
    - [ ] Create an `IntegratedNCA` that is conditioned by a global style vector (`w`).
    - [ ] Create an `IntegratedGenerator` that uses the `SharedBrain`.
- [ ] **Combined Training Loop:** Develop a `train_integrated.py` script where:
    - [ ] Both models are trained simultaneously.
    - [ ] A single `w` vector from the mapping network conditions both the NCA's growth and the StyleGAN's synthesis.
    - [ ] Gradients from both NCA and StyleGAN losses update the `SharedBrain`.
- [ ] **Multi-target Discriminator:** Update the discriminator to distinguish between real images, StyleGAN fakes, and NCA fakes.

## Priority 3: Visualization (In Progress)

- [x] **Web Server:** Set up a basic web server (e.g., Flask) in `ui/server.py`.
- [x] **Frontend Structure:** Create the basic HTML/CSS/JS structure for the visualization UI.
- [x] **Real-time Visualization:** Implement real-time visualization of the generated images from both models.
- [ ] **Latent Space Visualization:** Add a component to visualize the latent spaces.

## Priority 4: Refinements and Experiments

- [ ] **Progressive Growing:** Implement progressive growing for StyleGAN to generate high-resolution images (e.g., 256x256, 512x512).
- [ ] **Hyperparameter Tuning:** Tune the hyperparameters of the models and the training process.
- [ ] **Experiment Tracking:** Integrate a tool for tracking experiments (e.g., TensorBoard, W&B).
- [ ] **Model Architecture Improvements:** Explore and implement improvements to the model architectures. 