| Version | Date       | Description                                                                 | Changes Included                                                                                    |
| ------- | ---------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| v1.0.0  | 2025-10-04 | Initial complete working version with full training and inference pipeline. | - Modularized scripts for dataset preparation, model loading, training, inference, and API service. |
|         |            |                                                                             | - Integrated `config.json` for all configuration parameters.                                        |
|         |            |                                                                             | - Added logging functionality for reliability and debugging.                                        |
|         |            |                                                                             | - Support for CPU and GPU inference in a flexible way.                                              |
|         |            |                                                                             | - Configurations for DPO fine-tuning with LoRA on Unsloht Qwen3-0.6B 4-bit model.                   |
|         |            |                                                                             | - Added inference utility with adaptive token size based on VRAM availability.                      |
|         |            |                                                                             | - Added FastAPI server setup (`app.py` and `start_api.py`) for deployment.                          |
|         |            |                                                                             | - Included wandb integration for training tracking.                                                 |
|         |            |                                                                             | - Added README.md documenting configuration choices, workflow, and reasoning.                       |

Codebase State: Stable baseline for fine-tuning Fitness QA bot with Unsloht + LoRA + DPO.
Config State: Finalized config.json as above, containing dataset paths, training parameters, model configuration, inference settings, wandb settings, and logging configuration.
Key Highlights of This Version:

Fully modularized architecture for maintainability.

Flexible configuration-driven system.

Logging for reliability.

CPU/GPU compatible inference.

Optimized settings for small dataset (~4,200 training + 500 validation samples).

LoRA applied to only QKV and attention modules for efficiency.

wandb tracking enabled.

Production-ready API endpoint configuration.
