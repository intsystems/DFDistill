Data-Free Distillation
-------------------------

**Authors**: Ernest Nasyrov, Nikita Okhotnikov, Yuri Sapronov, Vladimir Solodkin

Description
--------------

This project focuses on implementing Data-Free Distillation in a simple and clear manner. 
Classical approaches to this problem perofrm distillation using logits, responses or hidden 
state from teacher obtained from data. However in some cases we cannot use the original data, 
and thus these methods become unapplicable. Our goal is to create a well-documented and 
efficient implementation for this complicated setting.

Algorithms Implemented
-------------------------

We plan to implement the following distillation techniques in our library:

- `Data-Free Knowledge Distillation using Top Layer Activation Statistics`
- `Data-Free Knowledge Distillation using Spectral Methods`
- `Data-Free Adversarial Distillation`
- `Data-Free Knowledge Transfer via DeepInversion`

Related Work
---------------

-  `Original paper <https://arxiv.org/pdf/1710.07535>`__
-  `Adversarial Approach <https://arxiv.org/pdf/1912.11006>`__
-  `Dreaming to Distill <https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.pdf>`__

Tech Stack
-------------

The project is implemented using:

- **PyPI**
- **PyTorch** for tensor computation and differentiation
- **Matplotlib** for plotting
- **Transformers** 
- **Neptune** for logging
- **Aquvitae** for distillation

Links
--------

-  `Project Documentation <./docs>`__

-  `Project Plan <https://github.com/intsystems/BMM/blob/main-24-25/projects.md>`__

-  `Blogpost <./DFDistill_Blogpost.pdf>`__
