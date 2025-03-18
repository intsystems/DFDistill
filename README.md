<div align="center">  
    <h1> Data-Free Distillation </h1>
</div>
<table>
    <tr>
        <td align="left"> <b> Title </b> </td>
        <td> DFDistill </td>
    </tr>
    <tr>
        <td align="left"> <b> Authors </b> </td>
        <td> Ernest Nasyrov, Nikita Okhotnikov, Yuri Sapronov, Vladimir Solodkin </td>
    </tr>
</table>

## Description

This project focuses on implementing Data-Free Distillation in a simple and clear manner. Classical approaches to this problem perofrm distillation using logits, responses or hidden state from teacher obtained from data. However in some cases we cannot use the original data, and thus these methods become unapplicable. Our goal is to create a well-documented and efficient implementation for this complicated setting.

## Algorithms Implemented

We plan to implement the following distillation techniques in our library:
- [ ] **Data-Free Knowledge Distillation using Top Layer Activation Statistics**
- [ ] **Data-Free Knowledge Distillation using Spectral Methods**
- [ ] **Data-Free Adversarial Distillation**
- [ ] **Data-Free Knowledge Transfer via DeepInversion**

## Related Work

- [Oridginal paper](https://arxiv.org/pdf/1710.07535)
- [Adversarial Approach](https://arxiv.org/pdf/1912.11006)
- [Dreaming to Distill](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.pdf)

## Tech Stack

The project is implemented using:

- **Python**
- **PyTorch** for tensor computation and differentiation
- **NumPy** for numerical computations
- **SciPy** for advanced mathematical functions
- **Jupyter Notebooks** for experimentation and visualization

You can install the required packages using pip:

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the cloned directory:
    ```bash
    cd <repository-directory>
    ```
3. Install the dependencies in editable mode:
    ```bash
    pip install -e ./
    ```
## Links

- [Project Documentation](./docs)
- [Project Plan](https://github.com/intsystems/BMM/blob/main-24-25/projects.md)
- [Blogpost](./DFDistill_Blogpost.pdf)
---
