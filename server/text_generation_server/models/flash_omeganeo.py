import math

import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoConfig, AutoTokenizer
from typing import Optional

from text_generation_server.models.cache_manager import BLOCK_SIZE
from text_generation_server.models.flash_mistral import (
    BaseFlashMistral,
    set_sliding_window,
)
from text_generation_server.models.custom_modeling.flash_omeganeo_modeling import (
    OmegaNeoForCausalLM,
)
# from text_generation_server.models.configuration.flash_omeganeo_configuration.py import OmegaNeoConfig
from transformers.configuration_utils import PretrainedConfig
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

tracer = trace.get_tracer(__name__)

class OmegaNeoConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class FlashOmegaNeo(BaseFlashMistral):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashOmegaNeo is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = OmegaNeoConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.use_medusa = use_medusa

        # Set context windows
        if config.sliding_window is not None:
            set_sliding_window(
                config.sliding_window, math.ceil(config.sliding_window / BLOCK_SIZE)
            )

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        model = OmegaNeoForCausalLM(config, weights)

        self.cuda_graphs = {}

        torch.distributed.barrier(group=self.process_group)
        super(BaseFlashMistral, self).__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=config.sliding_window,
        )
