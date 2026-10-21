# Third-party notices

## Reading Recognition in the Wild

Upstream project: https://github.com/facebookresearch/reading_in_the_wild

Seattle dataset and upstream code: distributed by Meta under CC BY-NC 4.0, according to the upstream README and LICENSE. The copied upstream license text is in `LICENSES/CC-BY-NC-4.0.txt`. Dataset-derived summaries in this release are secondary analyses; no endorsement by the dataset creators is implied.

Columbus dataset: distributed by Ohio State University. Its official Hugging Face dataset card declares `cc-by-nc-4.0`; its distributor and terms are separate from the Seattle distribution. No Columbus raw data is redistributed here.

The analysis uses the documented upstream default device-to-CPF transform and official measurement conventions. This release contains new analysis source plus derived scientific summaries and synthetic experiments, not a replacement distribution of the source dataset. Export changes include removing source identifiers and private runtime metadata from summaries, selecting source files, removing figure metadata, and making one real-trace example explicitly configured. Retained scientific numeric outputs are unchanged.

Required upstream attribution:

```bibtex
@inproceedings{yang25reading,
  title={Reading Recognition in the Wild},
  author={Charig Yang and Samiul Alam and Shakhrul Iman Siam and Michael Proulx and Lambert Mathias and Kiran Somasundaram and Luis Pesqueira and James Fort and Sheroze Sheriffdeen and Omkar Parkhi and Carl Ren and Mi Zhang and Yuning Chai and Richard Newcombe and Hyo Jin Kim},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://arxiv.org/abs/2505.24848}
}
```

These names identify the upstream dataset publication, not the authors of this anonymous analysis release. Preserve this attribution and the applicable noncommercial terms when sharing dataset-derived content. CC BY-NC 4.0 does not itself grant privacy or personality rights over recorded participants.

License reference: https://creativecommons.org/licenses/by-nc/4.0/
