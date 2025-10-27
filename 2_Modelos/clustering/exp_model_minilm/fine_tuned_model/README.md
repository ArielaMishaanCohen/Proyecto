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
- source_sentence: Andrew Cuomo Turns To Shockingly Racist Attacks On Zohran Mamdani
    In NYC Andrew Cuomo Turns To Shockingly Racist Attacks On Zohran Mamdani In NYC
  sentences:
  - The East Wing could fully be demolished soon, as preservationists urge caution
    The East Wing could fully be demolished soon, as preservationists urge caution
  - First high resolution structure of key herpes virus protein opens path to new
    antivirals First high resolution structure of key herpes virus protein opens path
    to new antivirals
  - Barbra Streisand Slams Latest Ridiculous Trump Move As Symbolic Of His Presidency
    Barbra Streisand Slams Latest Ridiculous Trump Move As Symbolic Of His Presidency
- source_sentence: Exclusive Trump Endorsed Massie Challenger Ed Gallrein Massie Not
    a Statesman Exclusive Trump Endorsed Massie Challenger Ed Gallrein Massie Not
    a Statesman
  sentences:
  - Coca Cola to Sell Cane Sugar Coke After Trump Said It s Just Better Coca Cola
    to Sell Cane Sugar Coke After Trump Said It s Just Better
  - Trump Hits Back Despite Ontario, Caught Red Handed, Suspending 54,000,000 Anti
    Tariff Ad Campaign Trump Hits Back Despite Ontario, Caught Red Handed, Suspending
    54,000,000 Anti Tariff Ad Campaign
  - PM implores school trust to install field ramp PM implores school trust to install
    field ramp
- source_sentence: Veterans, rural residents, older adults may lose food stamps due
    to Trump work requirements Veterans, rural residents, older adults may lose food
    stamps due to Trump work requirements
  sentences:
  - Death Row Appeal Claims Prosecutors Used Fat Shaming Evidence Unfairly Death Row
    Appeal Claims Prosecutors Used Fat Shaming Evidence Unfairly
  - Taylor Swift, LL Cool J, Kenny Loggins and David Byrne are among Songwriters Hall
    of Fame nominees Taylor Swift, LL Cool J, Kenny Loggins and David Byrne are among
    Songwriters Hall of Fame nominees
  - Report NYC Mayor Eric Adams to Endorse Andrew Cuomo for Mayor Report NYC Mayor
    Eric Adams to Endorse Andrew Cuomo for Mayor
- source_sentence: Fallout New Vegas 15th Anniversary Bundle Announced on Fallout
    Day Here Are the Contents Fallout New Vegas 15th Anniversary Bundle Announced
    on Fallout Day Here Are the Contents
  sentences:
  - Pool crunch coming for Prince George as aquatic centre set to close for 2 years
    Pool crunch coming for Prince George as aquatic centre set to close for 2 years
  - AU Deals Smash Hits, Dragon Hunts and Absolute Must Owns Light Up This Week s
    Sales AU Deals Smash Hits, Dragon Hunts and Absolute Must Owns Light Up This Week
    s Sales
  - Nearly 1 in 5 urinary tract infections linked to contaminated meat, study finds
    Nearly 1 in 5 urinary tract infections linked to contaminated meat, study finds
- source_sentence: Duffy warns of travel disruptions as air traffic controllers face
    missed paycheck Axios Duffy warns of travel disruptions as air traffic controllers
    face missed paycheck Axios
  sentences:
  - The Rules of Investing Just Changed Are You Ready? The Rules of Investing Just
    Changed Are You Ready?
  - Disgraced Prince Andrew Offered an Arabian Palace to Live in Luxury in Abu Dhabi,
    Away from the Mounting Pressures in the UK Over His Endless Scandals Disgraced
    Prince Andrew Offered an Arabian Palace to Live in Luxury in Abu Dhabi, Away from
    the Mounting Pressures in the UK Over His Endless Scandals
  - Broadway musicians reach tentative labor deal, averting a strike AP News Broadway
    musicians reach tentative labor deal, averting a strike AP News
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
      value: 0.8113207817077637
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
    'Duffy warns of travel disruptions as air traffic controllers face missed paycheck Axios Duffy warns of travel disruptions as air traffic controllers face missed paycheck Axios',
    'Broadway musicians reach tentative labor deal, averting a strike AP News Broadway musicians reach tentative labor deal, averting a strike AP News',
    'The Rules of Investing Just Changed Are You Ready? The Rules of Investing Just Changed Are You Ready?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5640, 0.4392],
#         [0.5640, 1.0000, 0.3495],
#         [0.4392, 0.3495, 1.0000]])
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
| **cosine_accuracy** | **0.8113** |

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
  | sentence_0                                                                                                                                                                                                                   | sentence_1                                                                                                                                                                                                                                                                                 | label            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Manipur marks Ningol Chakkouba amid hope heartbreak for IDPs Manipur marks Ningol Chakkouba amid hope heartbreak for IDPs</code>                                                                                       | <code>Arbaz Patel loses weight after reality show Rise and Fall says I wanted to reset physically and mentally Arbaz Patel loses weight after reality show Rise and Fall says I wanted to reset physically and mentally</code>                                                             | <code>0.9</code> |
  | <code>House Democrats request details on White House ballroom from President Trump Arden Farhi CBS News House Democrats request details on White House ballroom from President Trump Arden Farhi CBS News</code>             | <code>Top Democrats demand details of spy agencies role in boat strikes Washington Post Top Democrats demand details of spy agencies role in boat strikes Washington Post</code>                                                                                                           | <code>0.9</code> |
  | <code>The 2025 NAB Show New York Wraps Spotlighting the Future of Media, Storytelling and the Creator Economy The 2025 NAB Show New York Wraps Spotlighting the Future of Media, Storytelling and the Creator Economy</code> | <code>DOW CLASS ACTION ALERT Bragar Eagel Squire, P.C. Urgently Reminds Dow Investors of the October 28th Deadline in the Filed Class Action DOW CLASS ACTION ALERT Bragar Eagel Squire, P.C. Urgently Reminds Dow Investors of the October 28th Deadline in the Filed Class Action</code> | <code>0.9</code> |
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
- `disable_tqdm`: True
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
- `disable_tqdm`: True
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
| 1.0    | 39   | 0.7594                     |
| 1.2821 | 50   | 0.7736                     |
| 2.0    | 78   | 0.8113                     |


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