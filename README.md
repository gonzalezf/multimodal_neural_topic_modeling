# Multimodal Neural Topic Models
This repository contains the code for our paper **Neural Multimodal Topic Modeling: A Comprehensive Evaluation**, presented at LREC-COLING 2024.

## Abstract
Neural topic models can successfully find coherent and diverse topics in textual data. However, they are limited in dealing with multimodal datasets (e.g., images and text). This paper presents the first systematic and comprehensive evaluation of multimodal topic modeling for documents containing both text and images. In the process, we propose two novel topic modeling solutions and two novel evaluation metrics. Our evaluation on an unprecedentedly rich and diverse collection of datasets indicates that both of our models generate coherent and diverse topics. Nevertheless, the extent to which one method outperforms the other depends on the metrics and dataset combinations, suggesting further exploration of hybrid solutions in the future. Notably, our succinct human evaluation aligns with the outcomes determined by our proposed metrics. This alignment not only reinforces the credibility of our metrics but also highlights their potential application in guiding future multimodal topic modeling endeavors.

## Proposed Models

Our proposed models are based on the following frameworks:

- **Multimodal-ZeroShotTM:** A novel multimodal topic modeling algorithm based on ZeroShotTM (Bianchi et al., 2021b).
- **Multimodal-Contrast:** An adaptation specifically derived from M3L-Contrast (Zosa and Pivovarova, 2022a). While M3L-Contrast is a neural topic modeling technique tailored for analyzing datasets that are both multilingual and multimodal, Multimodal-Contrast focuses solely on multimodal data.



## How to Use This Repository

Demonstrations of how to use our models are available in the following notebooks:

- [Multimodal-Contrast Tutorial](Multimodal_Contrast/notebooks/Multimodal_Contrast_Tutorial.ipynb)
- [Multimodal-ZeroShotTM Tutorial](Multimodal_ZeroShotTM/notebooks/Multimodal_ZeroShotTM_tutorial.ipynb)


## Citation

Please cite our paper if you use our models or findings in your research.

```
@inproceedings{gonzalez-pizarro-carenini-2024-neural-multimodal,
    title = "Neural Multimodal Topic Modeling: A Comprehensive Evaluation",
    author = "Gonzalez-Pizarro, Felipe  and Carenini, Giuseppe",
    editor = "Calzolari, Nicoletta  and Kan, Min-Yen  and Hoste, Veronique  and Lenci, Alessandro  and Sakti, Sakriani  and Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1064",
    pages = "12159--12172",
    abstract = "Neural topic models can successfully find coherent and diverse topics in textual data. However, they are limited in dealing with multimodal datasets (e.g., images and text). This paper presents the first systematic and comprehensive evaluation of multimodal topic modeling of documents containing both text and images. In the process, we propose two novel topic modeling solutions and two novel evaluation metrics. Our evaluation on an unprecedentedly rich and diverse collection of datasets indicates that both of our models generate coherent and diverse topics. Nevertheless, the extent to which one method outperforms the other depends on the metrics and dataset combinations, suggesting further exploration of hybrid solutions in the future. Notably, our succinct human evaluation aligns with the outcomes determined by our proposed metrics. This alignment not only reinforces the credibility of our metrics but also highlights their potential application in guiding future multimodal topic modeling endeavors.",
}
```

## Contact

For any queries or further discussion regarding our work, feel free to contact us:

Felipe González-Pizarro  
felipeandresgonzalezpizarro[at]gmail[dot]com

**Thank you for your interest in our work on multimodal topic modeling!**
