# CR²: Consistency-Regularized Coopetitive Reasoning for Strategic Decision-Making in No-Press Diplomacy

This repository provides the supplementary code for the paper:

**CR²: Consistency-Regularized Coopetitive Reasoning for Strategic Decision-Making in No-Press Diplomacy**

The codebase is built on top of the Cicero / FairDiplomacy framework. We preserve the original project structure and add the implementation of the proposed consistency-regularized coopetitive reasoning method for strategic decision-making in no-press Diplomacy.

For more details about the original Cicero codebase, environment setup, model files, and general usage, please refer to [readme_cicero.md](readme_cicero.md).

---

## Method Overview

The overall method is illustrated in the following figure.

<p align="center">
  <img src="method%20V2.png" alt="CR² method overview" width="95%">
</p>

The PDF version of the method figure is also provided here:

[View method figure: `method V2.pdf`](method%20V2.pdf)

---

## Code

A brief orientation:

- The main implementation of the proposed CR² agent is located at:

```text
fairdiplomacy/agents/consistent_agent.py
```

- The corresponding agent configuration file is located at:

```text
conf/common/agents/consistent_agent.prototxt
```

- The runner file used for executing the consistency-aware agent logic is located at:

```text
fairdiplomacy/agents/consistent_runner.py
```

- The implementation follows the original FairDiplomacy agent interface and is designed to work within the existing Cicero configuration, rollout, and evaluation pipeline.

- The original Cicero / FairDiplomacy codebase is retained for compatibility with the underlying no-press Diplomacy environment, policy models, value models, configuration system, and evaluation scripts.

---

## Data and Logs

The related experimental logs and data are provided in:

```text
log_batch.zip
```

Users can extract this archive to inspect the corresponding evaluation records and related output files.

---

## Installation

This repository follows the installation procedure of the original Cicero codebase. A simplified setup is shown below.

```bash
# Clone the repository
git clone https://github.com/13jqq/Supplementary_for_CR_Diplomacy.git
cd Supplementary_for_CR_Diplomacy

# Create a conda environment
conda create --yes -n cr2_diplomacy python=3.7
conda activate cr2_diplomacy

# Install PyTorch and basic dependencies
conda install --yes pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
conda install --yes pybind11
conda install --yes go protobuf=3.19.1

# Install Python requirements
pip install -r requirements.txt

# Install local packages
pip install -e ./thirdparty/github/fairinternal/postman/nest/
pip install -e ./thirdparty/github/fairinternal/postman/postman/
pip install -e . -vv

# Build C++ / protobuf components
make
```

After installation, it is recommended to run:

```bash
make test_fast
```

For the full dependency list, compilation notes, model-file downloading instructions, and original usage examples, please see [readme_cicero.md](readme_cicero.md).

---

## Getting Started

The CR² implementation is organized as an agent module under the original FairDiplomacy agent directory. The main files are:

```text
fairdiplomacy/agents/consistent_agent.py
conf/common/agents/consistent_agent.prototxt
fairdiplomacy/agents/consistent_runner.py
```

The remaining configuration and execution pipeline follows the original Cicero / FairDiplomacy framework. Users may refer to [readme_cicero.md](readme_cicero.md) for instructions on running agents, configuring tasks, downloading model files, and launching evaluations.

---

## Repository Structure

```text
Supplementary_for_CR_Diplomacy/
├── README.md
├── readme_cicero.md
├── method V2.png
├── method V2.pdf
├── log_batch.zip
├── conf/
│   └── common/
│       └── agents/
│           └── consistent_agent.prototxt
├── fairdiplomacy/
│   └── agents/
│       ├── consistent_agent.py
│       └── consistent_runner.py
├── parlai_diplomacy/
├── thirdparty/
└── ...
```

---

## Notes

- This repository is provided as supplementary material for the CR² paper.
- The implementation is based on the original Cicero / FairDiplomacy codebase.
- The CR²-specific implementation is mainly located in `fairdiplomacy/agents/consistent_agent.py`, with the corresponding configuration in `conf/common/agents/consistent_agent.prototxt` and the runner logic in `fairdiplomacy/agents/consistent_runner.py`.
- Environment setup, model downloading, and most framework-level instructions follow the original Cicero repository and are documented in [readme_cicero.md](readme_cicero.md).

---

## License

This repository inherits the licensing structure of the original Cicero / FairDiplomacy codebase. Please refer to the original license files and [readme_cicero.md](readme_cicero.md) for details.
