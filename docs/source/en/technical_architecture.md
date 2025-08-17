<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Technical Architecture

This document provides a comprehensive overview of the Transformers library's technical architecture, design patterns, and implementation details. It serves as a technical reference for developers, researchers, and contributors who want to understand how Transformers works under the hood.

## 🏛️ Overall System Architecture

Transformers is built on a layered architecture that promotes modularity, extensibility, and framework interoperability. The system consists of several key layers:

```mermaid
graph TB
    subgraph "🎯 Application Layer"
        Apps[Applications & Scripts<br/>examples/, notebooks/]
        CLI[Command Line Tools<br/>transformers-cli]
    end
    
    subgraph "🚀 API Layer"
        Pipeline[Pipeline API<br/>High-level inference]
        Trainer[Trainer API<br/>Training & fine-tuning]
        Generate[Generation API<br/>Text generation]
    end
    
    subgraph "🔧 Core Layer"
        AutoClasses[Auto Classes<br/>Dynamic model loading]
        Models[Model Implementations<br/>360+ architectures]
        Processing[Preprocessing Components<br/>Tokenizers, Processors]
        Config[Configuration System<br/>Model parameters]
    end
    
    subgraph "🛠️ Infrastructure Layer"
        Utils[Utilities & Helpers<br/>Common functions]
        Integration[Integrations<br/>Framework adapters]
        Optimization[Optimization<br/>Performance enhancements]
    end
    
    subgraph "⚙️ Backend Layer"
        PyTorch[PyTorch Backend<br/>Primary implementation]
        TensorFlow[TensorFlow Backend<br/>Alternative implementation]
        JAX[JAX/Flax Backend<br/>High-performance implementation]
    end
    
    subgraph "📦 External Dependencies"
        HFHub[🤗 Hub<br/>Model repository]
        Accelerate[🤗 Accelerate<br/>Distributed training]
        Datasets[🤗 Datasets<br/>Data loading]
        Safetensors[Safetensors<br/>Secure serialization]
    end
    
    Apps --> Pipeline
    Apps --> Trainer
    CLI --> AutoClasses
    
    Pipeline --> Models
    Trainer --> Models
    Generate --> Models
    
    AutoClasses --> Models
    AutoClasses --> Processing
    AutoClasses --> Config
    
    Models --> Utils
    Processing --> Utils
    Config --> Utils
    
    Models --> PyTorch
    Models --> TensorFlow
    Models --> JAX
    
    Utils --> Integration
    Integration --> Optimization
    
    AutoClasses --> HFHub
    Trainer --> Accelerate
    Pipeline --> Datasets
    Models --> Safetensors
    
    style Pipeline fill:#e1f5fe
    style Models fill:#f3e5f5
    style PyTorch fill:#ffebee
    style HFHub fill:#e8f5e8
```

## 🧩 Core Components Architecture

### 1. Model Architecture Pattern

All models in Transformers follow a consistent three-component architecture:

```mermaid
classDiagram
    class PretrainedConfig {
        +model_type: str
        +hidden_size: int
        +num_attention_heads: int
        +from_pretrained()
        +save_pretrained()
        +to_dict()
    }
    
    class PreTrainedModel {
        +config: PretrainedConfig
        +forward()
        +from_pretrained()
        +save_pretrained()
        +generate()
        +get_input_embeddings()
        +set_input_embeddings()
    }
    
    class PreTrainedTokenizer {
        +vocab_size: int
        +encode()
        +decode()
        +tokenize()
        +from_pretrained()
        +save_pretrained()
    }
    
    PreTrainedModel --> PretrainedConfig : uses
    PreTrainedModel --> PreTrainedTokenizer : works with
    
    class BertConfig {
        +vocab_size: int
        +hidden_size: int
        +num_hidden_layers: int
        +intermediate_size: int
    }
    
    class BertModel {
        +embeddings: BertEmbeddings
        +encoder: BertEncoder  
        +pooler: BertPooler
        +forward()
    }
    
    class BertTokenizer {
        +wordpiece_tokenizer
        +encode()
        +decode()
    }
    
    BertConfig --|> PretrainedConfig
    BertModel --|> PreTrainedModel
    BertTokenizer --|> PreTrainedTokenizer
```

### 2. Auto Classes System

The Auto Classes provide a unified interface for loading models dynamically:

```mermaid
graph LR
    subgraph "Auto Classes"
        AutoConfig[AutoConfig<br/>Configuration loading]
        AutoModel[AutoModel<br/>Model loading]  
        AutoTokenizer[AutoTokenizer<br/>Tokenizer loading]
        AutoProcessor[AutoProcessor<br/>Processor loading]
    end
    
    subgraph "Registry System"
        ConfigMapping[CONFIG_MAPPING<br/>config_class → model_type]
        ModelMapping[MODEL_MAPPING<br/>model_class → config_class]
        TokenizerMapping[TOKENIZER_MAPPING<br/>tokenizer_class → config_class]
    end
    
    subgraph "Model Hub"
        HubFiles[config.json<br/>pytorch_model.bin<br/>tokenizer.json<br/>preprocessor_config.json]
    end
    
    AutoConfig --> ConfigMapping
    AutoModel --> ModelMapping
    AutoTokenizer --> TokenizerMapping
    
    ConfigMapping --> HubFiles
    ModelMapping --> HubFiles
    TokenizerMapping --> HubFiles
    
    style AutoConfig fill:#e3f2fd
    style AutoModel fill:#e8f5e8
    style AutoTokenizer fill:#fff3e0
    style HubFiles fill:#f3e5f5
```

## 🔄 Model Lifecycle & Data Flow

### 1. Model Loading Flow

```mermaid
sequenceDiagram
    participant User
    participant AutoModel
    participant Registry
    participant Hub
    participant ModelClass
    participant Config
    
    User->>AutoModel: from_pretrained("model_name")
    AutoModel->>Hub: Download config.json
    Hub-->>AutoModel: config.json
    AutoModel->>Config: Create config instance
    Config-->>AutoModel: config object
    AutoModel->>Registry: Look up model class
    Registry-->>AutoModel: model_class
    AutoModel->>Hub: Download model weights
    Hub-->>AutoModel: model weights
    AutoModel->>ModelClass: Initialize with config
    ModelClass-->>AutoModel: model instance
    AutoModel->>ModelClass: Load state dict
    AutoModel-->>User: Ready-to-use model
```

### 2. Training Pipeline Flow

```mermaid
graph TD
    subgraph "Data Preparation"
        Dataset[📊 Dataset<br/>Raw data]
        Tokenizer[🔤 Tokenizer<br/>Text → tokens]
        DataCollator[📦 Data Collator<br/>Batching & padding]
    end
    
    subgraph "Training Loop"
        Model[🧠 Model<br/>Forward pass]
        Loss[💥 Loss Function<br/>Calculate loss]
        Optimizer[⚡ Optimizer<br/>Update parameters]
        Scheduler[📈 Scheduler<br/>Learning rate]
    end
    
    subgraph "Monitoring & Control"
        Callbacks[🔔 Callbacks<br/>Custom logic]
        Logging[📝 Logging<br/>Metrics tracking]
        Checkpoints[💾 Checkpoints<br/>Save/restore]
    end
    
    subgraph "Distributed Training"
        DataParallel[📤 Data Parallel<br/>Multi-GPU]
        ModelParallel[🔄 Model Parallel<br/>Large models]
        DeepSpeed[🚀 DeepSpeed<br/>Zero redundancy]
    end
    
    Dataset --> Tokenizer
    Tokenizer --> DataCollator
    DataCollator --> Model
    Model --> Loss
    Loss --> Optimizer
    Optimizer --> Scheduler
    Scheduler --> Model
    
    Model --> Callbacks
    Callbacks --> Logging
    Logging --> Checkpoints
    
    Model --> DataParallel
    Model --> ModelParallel
    Model --> DeepSpeed
    
    style Model fill:#f3e5f5
    style Loss fill:#ffebee
    style Optimizer fill:#e8f5e8
```

### 3. Inference Pipeline Flow

```mermaid
graph LR
    subgraph "Input Processing"
        RawInput[📝 Raw Input<br/>Text/Image/Audio]
        Preprocessor[⚙️ Preprocessor<br/>Tokenize/Normalize]
        Tensors[🔢 Tensors<br/>Model inputs]
    end
    
    subgraph "Model Forward Pass"
        Embeddings[📊 Embeddings<br/>Input representation]
        Layers[🔄 Transformer Layers<br/>Attention + FFN]
        OutputHead[🎯 Output Head<br/>Task-specific]
    end
    
    subgraph "Output Processing"
        RawOutput[📈 Raw Outputs<br/>Logits/Predictions]
        Postprocessor[🎛️ Postprocessor<br/>Decode/Format]
        FinalOutput[✨ Final Output<br/>Human-readable]
    end
    
    RawInput --> Preprocessor
    Preprocessor --> Tensors
    Tensors --> Embeddings
    Embeddings --> Layers
    Layers --> OutputHead
    OutputHead --> RawOutput
    RawOutput --> Postprocessor
    Postprocessor --> FinalOutput
    
    style Preprocessor fill:#e3f2fd
    style Layers fill:#f3e5f5
    style Postprocessor fill:#e8f5e8
```

## 🔀 Multi-Framework Support

Transformers supports multiple ML frameworks through a unified interface:

```mermaid
graph TB
    subgraph "Unified Model Interface"
        ModelAPI[Common Model API<br/>forward(), generate(), etc.]
    end
    
    subgraph "Framework Implementations"
        PyTorchImpl[🔥 PyTorch Implementation<br/>modeling_*.py]
        TFImpl[🧮 TensorFlow Implementation<br/>modeling_tf_*.py] 
        FlaxImpl[⚡ JAX/Flax Implementation<br/>modeling_flax_*.py]
    end
    
    subgraph "Framework-Specific Features"
        PyTorchFeatures[🔥 PyTorch Features<br/>• Autograd<br/>• TorchScript<br/>• torch.compile]
        TFFeatures[🧮 TF Features<br/>• tf.function<br/>• SavedModel<br/>• TF Serving]
        FlaxFeatures[⚡ JAX Features<br/>• XLA compilation<br/>• Gradient transformation<br/>• Vectorization]
    end
    
    subgraph "Conversion Utilities"
        PyTorchToTF[PyTorch → TF<br/>Weight conversion]
        TFToPyTorch[TF → PyTorch<br/>Weight conversion]
        FlaxToPyTorch[Flax → PyTorch<br/>Weight conversion]
    end
    
    ModelAPI --> PyTorchImpl
    ModelAPI --> TFImpl
    ModelAPI --> FlaxImpl
    
    PyTorchImpl --> PyTorchFeatures
    TFImpl --> TFFeatures
    FlaxImpl --> FlaxFeatures
    
    PyTorchImpl <--> PyTorchToTF
    TFImpl <--> TFToPyTorch
    FlaxImpl <--> FlaxToPyTorch
    
    style ModelAPI fill:#e1f5fe
    style PyTorchImpl fill:#ffebee
    style TFImpl fill:#e8f5e8
    style FlaxImpl fill:#fff3e0
```

## 🚀 Performance Optimizations

### 1. Attention Optimizations

```mermaid
graph TD
    subgraph "Attention Mechanisms"
        StandardAttn[Standard Attention<br/>O(n²) complexity]
        FlashAttn[⚡ Flash Attention<br/>Memory-efficient]
        PagedAttn[📄 Paged Attention<br/>KV caching]
        SparseAttn[🕸️ Sparse Attention<br/>Reduced complexity]
    end
    
    subgraph "Hardware Acceleration"
        CUDA[🔥 CUDA Kernels<br/>GPU optimization]
        ROCm[🏔️ ROCm<br/>AMD GPU support]
        Metal[🍎 Metal<br/>Apple Silicon]
        CPU[🖥️ CPU Optimizations<br/>SIMD instructions]
    end
    
    subgraph "Framework Integration"
        SDPABackend[SDPA Backend<br/>PyTorch 2.0+]
        CustomKernels[Custom Kernels<br/>Triton/CUDA]
        CompileMode[torch.compile<br/>JIT compilation]
    end
    
    StandardAttn --> FlashAttn
    StandardAttn --> PagedAttn
    StandardAttn --> SparseAttn
    
    FlashAttn --> CUDA
    FlashAttn --> ROCm
    FlashAttn --> Metal
    StandardAttn --> CPU
    
    FlashAttn --> SDPABackend
    FlashAttn --> CustomKernels
    FlashAttn --> CompileMode
    
    style FlashAttn fill:#e8f5e8
    style CUDA fill:#ffebee
    style SDPABackend fill:#e3f2fd
```

### 2. Memory Management

```mermaid
graph LR
    subgraph "Memory Strategies"
        GradientCheckpointing[📦 Gradient Checkpointing<br/>Trade compute for memory]
        ModelSharding[🔄 Model Sharding<br/>Distribute across devices]
        OffloadingCPU[💾 CPU Offloading<br/>Move to system memory]
        QuantizationFP16[🔢 FP16/BF16<br/>Half precision]
    end
    
    subgraph "Advanced Techniques"
        DeepSpeedZeRO[🚀 DeepSpeed ZeRO<br/>Zero redundancy optimizer]
        FSDPFullyShard[🔗 FSDP<br/>Fully sharded data parallel]
        PipelineParallel[🔄 Pipeline Parallel<br/>Layer distribution]
    end
    
    subgraph "KV Cache Management"
        StaticCache[📊 Static KV Cache<br/>Pre-allocated]
        DynamicCache[🔄 Dynamic KV Cache<br/>Growing cache]
        SlidingWindow[🪟 Sliding Window<br/>Limited attention span]
    end
    
    GradientCheckpointing --> DeepSpeedZeRO
    ModelSharding --> FSDPFullyShard
    OffloadingCPU --> PipelineParallel
    
    StaticCache --> DynamicCache
    DynamicCache --> SlidingWindow
    
    style DeepSpeedZeRO fill:#e8f5e8
    style FSDPFullyShard fill:#e3f2fd
    style DynamicCache fill:#fff3e0
```

## 🌐 Ecosystem Integration

### 1. Training Framework Integration

```mermaid
graph TB
    subgraph "Core Transformers"
        TransformersModels[🤗 Transformers Models<br/>Model definitions]
        TransformersTrainer[🤗 Transformers Trainer<br/>Basic training loop]
    end
    
    subgraph "Specialized Training Frameworks"
        Axolotl[🪓 Axolotl<br/>Configuration-driven training]
        Unsloth[🦥 Unsloth<br/>Fast fine-tuning]
        TRL[🎭 TRL<br/>Reinforcement Learning]
        PEFT[📎 PEFT<br/>Parameter-efficient fine-tuning]
    end
    
    subgraph "Distributed Training"
        Accelerate[🚀 Accelerate<br/>Multi-GPU/Multi-node]
        DeepSpeed[💨 DeepSpeed<br/>Large model training]
        FairScale[⚖️ FairScale<br/>PyTorch scaling]
        ColossalAI[🏛️ Colossal-AI<br/>Large-scale training]
    end
    
    subgraph "Cloud & MLOps"
        SageMaker[☁️ AWS SageMaker<br/>Managed training]
        Weights_Biases[📊 Weights & Biases<br/>Experiment tracking]
        MLflow[🌊 MLflow<br/>ML lifecycle management]
    end
    
    TransformersModels --> Axolotl
    TransformersModels --> Unsloth
    TransformersModels --> TRL
    TransformersModels --> PEFT
    
    TransformersTrainer --> Accelerate
    TransformersTrainer --> DeepSpeed
    TransformersTrainer --> FairScale
    TransformersTrainer --> ColossalAI
    
    Axolotl --> SageMaker
    TransformersTrainer --> Weights_Biases
    TransformersTrainer --> MLflow
    
    style TransformersModels fill:#f3e5f5
    style Accelerate fill:#e8f5e8
    style SageMaker fill:#e3f2fd
```

### 2. Inference Engine Integration

```mermaid
graph TB
    subgraph "Model Definition"
        HFModels[🤗 Transformers Models<br/>Standard model definitions]
    end
    
    subgraph "High-Throughput Inference"
        vLLM[⚡ vLLM<br/>Fast LLM serving]
        TGI[🚀 Text Generation Inference<br/>Production serving]
        SGLang[🔤 SGLang<br/>Structured generation]
        TensorRT[🏎️ TensorRT-LLM<br/>NVIDIA optimization]
    end
    
    subgraph "Edge Deployment"
        ONNX[📦 ONNX<br/>Cross-platform inference]
        TorchScript[🔥 TorchScript<br/>PyTorch deployment]
        TFLite[📱 TensorFlow Lite<br/>Mobile inference]
        CoreML[🍎 Core ML<br/>Apple deployment]
    end
    
    subgraph "Specialized Backends"
        LlamaCpp[🦙 llama.cpp<br/>CPU-optimized inference]
        MLX[🍎 MLX<br/>Apple Silicon]
        Candle[🕯️ Candle<br/>Rust implementation]
        GGML[⚡ GGML<br/>Quantized inference]
    end
    
    HFModels --> vLLM
    HFModels --> TGI
    HFModels --> SGLang
    HFModels --> TensorRT
    
    HFModels --> ONNX
    HFModels --> TorchScript
    HFModels --> TFLite
    HFModels --> CoreML
    
    HFModels --> LlamaCpp
    HFModels --> MLX
    HFModels --> Candle
    HFModels --> GGML
    
    style HFModels fill:#f3e5f5
    style vLLM fill:#e8f5e8
    style ONNX fill:#e3f2fd
    style LlamaCpp fill:#fff3e0
```

## 📁 Code Organization

### Directory Structure

```
transformers/
├── 📁 src/transformers/          # Main library code
│   ├── 📁 models/               # Model implementations (360+ models)
│   │   ├── 📁 bert/            # BERT model files
│   │   │   ├── configuration_bert.py
│   │   │   ├── modeling_bert.py
│   │   │   ├── modeling_tf_bert.py
│   │   │   ├── modeling_flax_bert.py
│   │   │   └── tokenization_bert.py
│   │   └── ...                 # Other model directories
│   ├── 📁 pipelines/           # High-level Pipeline API
│   ├── 📁 generation/          # Text generation utilities
│   ├── 📁 quantizers/          # Model quantization
│   └── 📄 *.py                 # Core utilities and base classes
├── 📁 tests/                   # Comprehensive test suite
├── 📁 examples/                # Example scripts and notebooks
├── 📁 docs/                    # Documentation source
└── 📁 utils/                   # Development utilities
```

### Model Implementation Pattern

Each model follows a consistent file structure:

```mermaid
graph LR
    subgraph "Model Directory"
        Config[configuration_*.py<br/>Model configuration]
        PyTorchModel[modeling_*.py<br/>PyTorch implementation]
        TFModel[modeling_tf_*.py<br/>TensorFlow implementation]
        FlaxModel[modeling_flax_*.py<br/>JAX/Flax implementation]
        Tokenizer[tokenization_*.py<br/>Tokenizer implementation]
        Processor[processing_*.py<br/>Multi-modal processor]
    end
    
    subgraph "Auto Registration"
        AutoMapping[__init__.py<br/>Register in Auto classes]
    end
    
    Config --> PyTorchModel
    Config --> TFModel
    Config --> FlaxModel
    PyTorchModel --> Tokenizer
    TFModel --> Tokenizer
    FlaxModel --> Tokenizer
    Processor --> Config
    
    PyTorchModel --> AutoMapping
    TFModel --> AutoMapping
    FlaxModel --> AutoMapping
    Tokenizer --> AutoMapping
    
    style Config fill:#e3f2fd
    style PyTorchModel fill:#ffebee
    style TFModel fill:#e8f5e8
    style FlaxModel fill:#fff3e0
```

## 🔧 Development Patterns

### 1. Modular vs. Copied Code

Transformers uses two main approaches for code organization:

```mermaid
graph TD
    subgraph "Modular Approach"
        ModularFile[modular_*.py<br/>Compose from base classes]
        BaseClasses[Base transformer layers<br/>Reusable components]
        AutoGenerated[modeling_*.py<br/>Auto-generated full file]
    end
    
    subgraph "Copied Code Approach"
        BaseImplementation[Base implementation<br/>e.g., BERT layers]
        CopiedFunction[# Copied from transformers.models.bert<br/>Explicit copy comments]
        VariantImplementation[Variant implementation<br/>e.g., RoBERTa layers]
    end
    
    subgraph "Style Tools"
        MakeFixup[make fixup<br/>Update copies and generate modular files]
    end
    
    ModularFile --> BaseClasses
    BaseClasses --> AutoGenerated
    
    BaseImplementation --> CopiedFunction
    CopiedFunction --> VariantImplementation
    
    ModularFile --> MakeFixup
    CopiedFunction --> MakeFixup
    MakeFixup --> AutoGenerated
    MakeFixup --> VariantImplementation
    
    style ModularFile fill:#e8f5e8
    style BaseImplementation fill:#e3f2fd
    style MakeFixup fill:#fff3e0
```

### 2. Testing Strategy

```mermaid
graph TB
    subgraph "Test Categories"
        CommonTests[Common Tests<br/>Base functionality]
        ModelTests[Model-Specific Tests<br/>Architecture validation]
        IntegrationTests[Integration Tests<br/>End-to-end workflows]
        SlowTests[Slow Tests<br/>Large model validation]
    end
    
    subgraph "Test Infrastructure"
        TestUtils[Test Utilities<br/>Helper functions]
        Fixtures[Test Fixtures<br/>Sample data]
        CI[CI Pipeline<br/>Automated testing]
    end
    
    subgraph "Coverage Areas"
        ModelForward[Forward Pass<br/>Input/output validation]
        ModelBackward[Backward Pass<br/>Gradient computation]
        Serialization[Save/Load<br/>Model persistence]
        AutoIntegration[Auto Classes<br/>Dynamic loading]
    end
    
    CommonTests --> ModelTests
    ModelTests --> IntegrationTests
    IntegrationTests --> SlowTests
    
    CommonTests --> TestUtils
    TestUtils --> Fixtures
    Fixtures --> CI
    
    ModelForward --> ModelBackward
    ModelBackward --> Serialization
    Serialization --> AutoIntegration
    
    style CommonTests fill:#e8f5e8
    style CI fill:#e3f2fd
    style ModelForward fill:#fff3e0
```

## 🔍 Advanced Features

### 1. Dynamic Module Loading

Transformers supports loading custom models from remote repositories:

```mermaid
sequenceDiagram
    participant User
    participant AutoModel
    participant Hub
    participant LocalCache
    participant DynamicModule
    
    User->>AutoModel: from_pretrained("custom/model", trust_remote_code=True)
    AutoModel->>Hub: Check for modeling_*.py files
    Hub-->>AutoModel: Remote code files
    AutoModel->>LocalCache: Cache remote code
    LocalCache-->>AutoModel: Local paths
    AutoModel->>DynamicModule: Import remote modules
    DynamicModule-->>AutoModel: Custom model class
    AutoModel-->>User: Custom model instance
```

### 2. Quantization Support

```mermaid
graph LR
    subgraph "Quantization Methods"
        FP16[Half Precision<br/>FP16/BF16]
        INT8[8-bit Quantization<br/>LLM.int8()]
        INT4[4-bit Quantization<br/>QLoRA/GPTQ]
        AWQ[Activation-aware<br/>AWQ]
    end
    
    subgraph "Quantization Libraries"
        BitsAndBytes[🔢 bitsandbytes<br/>CUDA quantization]
        AutoGPTQ[⚡ auto-gptq<br/>GPTQ implementation]
        AutoAWQ[🔧 autoawq<br/>AWQ implementation]
        Optimum[🚀 Optimum<br/>Intel/ONNX quantization]
    end
    
    subgraph "Hardware Targets"
        GPU[🔥 GPU<br/>CUDA/ROCm]
        CPU[🖥️ CPU<br/>Intel/AMD]
        Mobile[📱 Mobile<br/>ARM/Edge]
    end
    
    FP16 --> BitsAndBytes
    INT8 --> BitsAndBytes
    INT4 --> AutoGPTQ
    AWQ --> AutoAWQ
    
    BitsAndBytes --> GPU
    AutoGPTQ --> GPU
    AutoAWQ --> GPU
    Optimum --> CPU
    Optimum --> Mobile
    
    style INT4 fill:#e8f5e8
    style BitsAndBytes fill:#e3f2fd
    style GPU fill:#ffebee
```

## 📚 Documentation & Resources

For more detailed information, refer to:

- [Philosophy](./philosophy): Core design principles and goals
- [Models](./models): How to load and use models
- [Pipeline Tutorial](./pipeline_tutorial): High-level API usage
- [Training](./training): Fine-tuning and training guides
- [Performance](./llm_optims): Optimization techniques
- [Contributing](./contributing): How to contribute to the library

This technical architecture guide provides the foundation for understanding how Transformers works internally. The modular design, consistent patterns, and extensive ecosystem integration make it a powerful framework for both research and production use cases.