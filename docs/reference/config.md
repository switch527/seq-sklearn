# Configuration reference

pydantic v2 config schemas rendered as field tables (type, default,
constraint, validators) via `autodoc-pydantic`. These are the
authoritative source for what every config field does and what
values are accepted.

## Top-level configs

```{eval-rst}
.. autopydantic_model:: seq_sklearn.TFTConfig
   :inherited-members: BaseModel

.. autopydantic_model:: seq_sklearn.TabularToSequenceConfig
   :inherited-members: BaseModel
```

## Training-side blocks

```{eval-rst}
.. autopydantic_model:: seq_sklearn.config.scheduler.SchedulerConfig
   :inherited-members: BaseModel

.. autopydantic_model:: seq_sklearn.config.optimizer.OptimizerConfig
   :inherited-members: BaseModel

.. autopydantic_model:: seq_sklearn.config.loss.LossConfig
   :inherited-members: BaseModel

.. autopydantic_model:: seq_sklearn.config.sampler.SamplerConfig
   :inherited-members: BaseModel
```
