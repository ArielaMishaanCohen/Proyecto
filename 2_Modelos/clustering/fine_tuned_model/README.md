---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:621
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Takaichi said to arrange phone call with Trump on Saturday Takaichi
    said to arrange phone call with Trump on Saturday
  sentences:
  - Bigg Boss 19 Farrhana Bhatt opens up about her strained relationship with her
    father says he is not a good father, my relatives threatened me for making videos
    Bigg Boss 19 Farrhana Bhatt opens up about her strained relationship with her
    father says he is not a good father, my relatives threatened me for making videos
  - Portland Trail Blazers coach Chauncey Billups and Miami Heat player Terry Rozier
    arrested BBC Portland Trail Blazers coach Chauncey Billups and Miami Heat player
    Terry Rozier arrested BBC
  - It s important to liberate Venezuela Congressional Republicans cheer Trump s offensive
    It s important to liberate Venezuela Congressional Republicans cheer Trump s offensive
- source_sentence: Aero News Quote of the Day 10.21.25 Aero News Quote of the Day
    10.21.25
  sentences:
  - Airborne Flight Training 10.23.25 PanAm Back?, Spirit Cuts, Affordable Expo Airborne
    Flight Training 10.23.25 PanAm Back?, Spirit Cuts, Affordable Expo
  - Alberta teen faces new charges after terrorism peace bond for alleged online extremism
    ties Alberta teen faces new charges after terrorism peace bond for alleged online
    extremism ties
  - The Fourth Edition of The MICHELIN Guide Abu Dhabi Has Been Launched The Fourth
    Edition of The MICHELIN Guide Abu Dhabi Has Been Launched
- source_sentence: Over 6,000 voters deleted for Rs 80 each in Aland Karnataka SIT
    Over 6,000 voters deleted for Rs 80 each in Aland Karnataka SIT
  sentences:
  - The Single Most Important AI Chart of the Decade The Single Most Important AI
    Chart of the Decade
  - LeBron James name surfaces in NBA mafia betting probe after private injury status
    allegedly leaked by former teammate LeBron James name surfaces in NBA mafia betting
    probe after private injury status allegedly leaked by former teammate
  - VRA Investor News If You Have Suffered Losses in Vera Bradley, Inc. NASDAQ VRA
    , You Are Encouraged to Contact The Rosen Law Firm About Your Rights VRA Investor
    News If You Have Suffered Losses in Vera Bradley, Inc. NASDAQ VRA , You Are Encouraged
    to Contact The Rosen Law Firm About Your Rights
- source_sentence: New York City Mayor Eric Adams backs Andrew Cuomo in the race to
    succeed him Katherine Koretski NBC News New York City Mayor Eric Adams backs Andrew
    Cuomo in the race to succeed him Katherine Koretski NBC News
  sentences:
  - Hydrothermal vents may have triggered early molecular chemistry on ancient Earth
    Hydrothermal vents may have triggered early molecular chemistry on ancient Earth
  - New book details infighting behind Trump s obviously unqualified cabinet picks
    David Smith The Guardian New book details infighting behind Trump s obviously
    unqualified cabinet picks David Smith The Guardian
  - Airborne Flight Training 10.23.25 PanAm Back?, Spirit Cuts, Affordable Expo Airborne
    Flight Training 10.23.25 PanAm Back?, Spirit Cuts, Affordable Expo
- source_sentence: Securities Fraud Investigation Into Western Alliance Bancorporation
    WAL Continues Investors Who Lost Money Urged To Contact Glancy Prongay Murray
    LLP, a Leading Securities Fraud Law Firm Securities Fraud Investigation Into Western
    Alliance Bancorporation WAL Continues Investors Who Lost Money Urged To Contact
    Glancy Prongay Murray LLP, a Leading Securities Fraud Law Firm
  sentences:
  - Colorado snow lovers, rejoice A Basin opens for the season Sunday Colorado snow
    lovers, rejoice A Basin opens for the season Sunday
  - Telix Pharmaceuticals Limited Investigated by the Portnoy Law Firm Telix Pharmaceuticals
    Limited Investigated by the Portnoy Law Firm
  - RCI HOSPITALITY CLASS ACTION REMINDER Bragar Eagel Squire, P.C. Reminds RCI Stockholders
    of the Filed Class Action Lawsuit and Urges Investors to Contact the Firm Before
    November 20th RCI HOSPITALITY CLASS ACTION REMINDER Bragar Eagel Squire, P.C.
    Reminds RCI Stockholders of the Filed Class Action Lawsuit and Urges Investors
    to Contact the Firm Before November 20th
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: validation
      type: validation
    metrics:
    - type: cosine_accuracy
      value: 0.7877358198165894
      name: Cosine Accuracy
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Securities Fraud Investigation Into Western Alliance Bancorporation WAL Continues Investors Who Lost Money Urged To Contact Glancy Prongay Murray LLP, a Leading Securities Fraud Law Firm Securities Fraud Investigation Into Western Alliance Bancorporation WAL Continues Investors Who Lost Money Urged To Contact Glancy Prongay Murray LLP, a Leading Securities Fraud Law Firm',
    'RCI HOSPITALITY CLASS ACTION REMINDER Bragar Eagel Squire, P.C. Reminds RCI Stockholders of the Filed Class Action Lawsuit and Urges Investors to Contact the Firm Before November 20th RCI HOSPITALITY CLASS ACTION REMINDER Bragar Eagel Squire, P.C. Reminds RCI Stockholders of the Filed Class Action Lawsuit and Urges Investors to Contact the Firm Before November 20th',
    'Colorado snow lovers, rejoice A Basin opens for the season Sunday Colorado snow lovers, rejoice A Basin opens for the season Sunday',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5628, 0.3646],
#         [0.5628, 1.0000, 0.3125],
#         [0.3646, 0.3125, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Dataset: `validation`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| **cosine_accuracy** | **0.7877** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 621 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 621 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                         |
  | details | <ul><li>min: 6 tokens</li><li>mean: 36.68 tokens</li><li>max: 94 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 37.28 tokens</li><li>max: 94 tokens</li></ul> | <ul><li>min: 0.9</li><li>mean: 0.9</li><li>max: 0.9</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                               | sentence_1                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Europe s new Sentinel 4 mission delivers first look at hourly air pollution maps Europe s new Sentinel 4 mission delivers first look at hourly air pollution maps</code>                                                                                                                           | <code>ABB wins Canadian climate satellite instrument contract ABB wins Canadian climate satellite instrument contract</code>                                                                                                                                                                                             | <code>0.9</code> |
  | <code>Game on Jack Smith offers to testify in public hearing over his investigation into Donald Trump Game on Jack Smith offers to testify in public hearing over his investigation into Donald Trump</code>                                                                                             | <code>Who are AI browsers for? Who are AI browsers for?</code>                                                                                                                                                                                                                                                           | <code>0.9</code> |
  | <code>SEMLER CLASS ACTION REMINDER Bragar Eagel Squire, P.C. Urges Semler Scientific Investors to Contact the Firm Before the October 28th Deadline SEMLER CLASS ACTION REMINDER Bragar Eagel Squire, P.C. Urges Semler Scientific Investors to Contact the Firm Before the October 28th Deadline</code> | <code>GBank Financial Holdings Inc. Announces Third Quarter 2025 Quarterly Earnings Call Scheduled for Wednesday, October 29th, at 10 00 A.M., Pacific Time GBank Financial Holdings Inc. Announces Third Quarter 2025 Quarterly Earnings Call Scheduled for Wednesday, October 29th, at 10 00 A.M., Pacific Time</code> | <code>0.9</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | validation_cosine_accuracy |
|:------:|:----:|:--------------------------:|
| 1.0    | 39   | 0.7547                     |
| 1.2821 | 50   | 0.7877                     |


### Framework Versions
- Python: 3.10.2
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.7.1
- Accelerate: 1.10.1
- Datasets: 4.3.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->