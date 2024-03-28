# Generative AI Models 

## Open Source Models 

<br>

### Text Generation 

|  Model  | Created By | Size |  Description |    Link  |
| ------- | ----------- | --------- | ------ | ----- | 
| Mistral 7B | Mistral AI | 7B | The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. Mistral-7B-v0.1 outperforms Llama 2 13B on most benchmarks. | [Github](https://github.com/mistralai/mistral-src) [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1) [Blog](https://mistral.ai/news/announcing-mistral-7b/) |
| Mistral 7B v2 | Mistral AI | 7B | Mistral 7B v2 has the following changes compared to Mistral 7B:- 32k context window (vs 8k context in v0.1), Rope-theta = 1e6, No Sliding-Window Attention. | [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | 
| Mixtral 8x7B | Mistral AI | Active 12B | Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks. | [Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) [Blog](https://mistral.ai/news/mixtral-of-experts/) | 
| Gemma | Google | 2B 7B | Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights, pre-trained variants, and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. | [Hugging Face](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)  [Github](https://github.com/google-deepmind/gemma) [Blog](https://blog.google/technology/developers/gemma-open-models/) | 
| Grok 1.0 | xAI | 314B |  | [Github](https://github.com/xai-org/grok-1) [Hugging Face](https://huggingface.co/xai-org/grok-1) | 
| Llama 2 | Meta AI | 7B 13B 70B | Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety. | [Hugging Face](https://huggingface.co/meta-llama) [Github](https://github.com/meta-llama/llama) [Blog](https://ai.meta.com/blog/llama-2/) | 
| Dolly v2 | Databricks | 3B 7B 12B | Dolly v2 is a causal language model created by Databricks that is derived from EleutherAI's Pythia-12b and fine-tuned on a ~15K record instruction corpus. | [Github](https://github.com/databrickslabs/dolly) | 
| DBRX | Databricks | 132B Active 36B | DBRX is a transformer-based decoder-only large language model (LLM) that was trained using next-token prediction. It uses a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input. It was pre-trained on 12T tokens of text and code data. Compared to other open MoE models like Mixtral-8x7B and Grok-1, DBRX is fine-grained, meaning it uses a larger number of smaller experts. DBRX has 16 experts and chooses 4, while Mixtral-8x7B and Grok-1 have 8 experts and choose 2. This provides 65x more possible combinations of experts which improves model quality. | [Hugging Face](https://huggingface.co/collections/databricks/dbrx-6601c0852a0cdd3c59f71962) [Github](https://github.com/databricks/dbrx) [Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | 
| Command-R | Cohere | 35B | Command-R is a research release of a 35 billion parameter highly performant generative model. Command-R is a large language model with open weights optimized for a variety of use cases including reasoning, summarization, and question answering. Command-R has the capability for multilingual generation evaluated in 10 languages and highly performant RAG capabilities. | [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01) | 
| Qwen1.5 | Alibaba | 7B 14B 72B |  |  | [Hugging Face](https://huggingface.co/Qwen) [Github](https://github.com/QwenLM/Qwen) | 
| Vicuna v1.5 | Lysms | 7B 13B  |  | [Hugging Face Vicuna 7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) [Hugging Face Vicuna 13B](https://huggingface.co/lmsys/vicuna-13b-v1.5) | 
| Phi 2 | Microsoft | 2.7B | | [Hugging Face](https://huggingface.co/microsoft/phi-2) |
| Orca 2 | Microsoft | 7B 13B | | [Hugging Face](https://huggingface.co/collections/microsoft/orca-65bbeef1980f5719cccc89a3) |
| Smaug | Abacus AI | 34B 72B | | [Hugging Face](https://huggingface.co/abacusai) | 
| MPT | Mosaicml | 1B 7B 30B | MPT is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. These models use a modified transformer architecture optimized for efficient training and inference. These architectural changes include performance-optimized layer implementations and the elimination of context length limits by replacing positional embeddings with Attention with Linear Biases (ALiBi). | [Hugging Face](https://huggingface.co/collections/mosaicml/mpt-6564f3d9e5aac326bfa22def) | 
| Falcon | TLL | 7B 40B 180B | Falcon is a 7B/40B/180B parameters causal decoder-only models built by TII and trained on  1,000B/1,500B/3,500B tokens of RefinedWeb enhanced with curated corpora. | [Hugging Face](https://huggingface.co/tiiuae) | 
| DeciLM | DeciAI | 6B 7B | DeciLM is a decoder-only text generation model. With support for an 8K-token sequence length, this highly efficient model uses variable Grouped-Query Attention (GQA) to achieve a superior balance between accuracy and computational efficiency. | [Hugging Face](https://huggingface.co/collections/Deci/decilm-models-65a7fb6a65e4f1a5eb14917a) |
| BERT | Google | 110M to 350M | | [GitHub](https://github.com/google-research/bert) | 
| Olmo | AllenAI | 1B 7B | | [Hugging Face](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) [Github](https://github.com/allenai/OLMo) | 
| Openchat3.5 | Openchat | 7B | Openchat2.5 is the best performing 7B LLM. | [Hugging Face](https://huggingface.co/collections/openchat/openchat-65110500e14eeb01d4888806) [Github](https://github.com/imoneoi/openchat) |
| Bloom | BigScience | | BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text data using industrial-scale computational resources. | [Hugging Face](https://huggingface.co/bigscience/bloom) |


<br>

### Code Generation

|  Model  | Created By | Size | Description |    Link  |
| ------- | ----------- | --------- | ------ | ----- | 
| Codellama | Meta AI | 7B 13B 34B 70B | Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. | [Github](https://github.com/meta-llama/codellama) [Hugging Face](https://huggingface.co/codellama) | 
| Starcoder | BigCode | 15.5B | The StarCoder models are 15.5B parameter models trained on 80+ programming languages from The Stack (v1.2), with opt-out requests excluded. The model uses Multi Query Attention, a context window of 8192 tokens, and was trained using the Fill-in-the-Middle objective on 1 trillion tokens. | [Hugging Face](https://huggingface.co/bigcode/starcoder) [Github](https://github.com/bigcode-project/starcoder) | 
| Starcoder2 | BigCode | 3B 7B 15B | StarCoder2-15B model is a 15B parameter model trained on 600+ programming languages from The Stack v2, with opt-out requests excluded. The model uses Grouped Query Attention, a context window of 16,384 tokens with a sliding window attention of 4,096 tokens, and was trained using the Fill-in-the-Middle objective on 4+ trillion tokens. | [Hugging Face](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a) [GitHub](https://github.com/bigcode-project/starcoder2) | 
| DeciCoder | DeciAI | 1B 6B | DeciCoder are decoder-only code completion models trained on the Python, Java, and Javascript subsets of Starcoder Training Dataset. The model uses Grouped Query Attention and has a context window of 2048 tokens. It was trained using a Fill-in-the-Middle training objective. | [Hugging Face](https://huggingface.co/collections/Deci/decicoder-models-65a7faf617d869bb743f1766) |  
| Stable Code | Stability AI | 3B | | [Hugging Face](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650) [Github](https://github.com/Stability-AI/StableCode) | 
| SqlCoder | DefogAI | 2B 34B 70B | | [Hugging Face](https://huggingface.co/defog) [Github](https://github.com/defog-ai/sqlcoder) | 
| SqlCoder2 | DefogAI | 2B 15B | | [Hugging Face](https://huggingface.co/defog) [Github](https://github.com/defog-ai/sqlcoder) | 
| DeepSeek Coder | Deepseek AI | 1B to 33B | Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. Various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and a extra fill-in-the-blank task, to support project-level code completion and infilling. For coding capabilities, Deepseek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks. | [Hugging Face](https://huggingface.co/collections/deepseek-ai/deepseek-coder-65f295d7d8a0a29fe39b4ec4) [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) | 
| Codegen2 | Salesforce | 1B 7B 16B | |  [Hugging Face](https://huggingface.co/Salesforce/codegen2-16B) [Github](https://github.com/salesforce/CodeGen) | 


<br>

### Image Generation

|  Model  | Created By | Description |    Link  |
| ------- | ----------- | --------- | ------ | 
| Stable Diffusion2 | Stability AI  |     |  | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2) [Github](https://github.com/Stability-AI/stablediffusion) |
| SDXL-turbo | Stability AI | SDXL-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation. SDXL-Turbo is a distilled version of SDXL 1.0, trained for real-time synthesis. SDXL-Turbo is based on a novel training method called Adversarial Diffusion Distillation (ADD). | [Hugging Face](https://huggingface.co/stabilityai/sdxl-turbo) | 
| Stable Cascade | Stability AI | Stable Cascade is a diffusion model trained to generate images given a text prompt. It  is built upon the Würstchen architecture and its main difference to other models like Stable Diffusion is that it is working at a much smaller latent space. |  [Hugging Face](https://huggingface.co/stabilityai/stable-cascade) [Github](https://github.com/Stability-AI/StableCascade) |
| DeciDiffusion v2.0 | DeciAI | DeciDiffusion 2.0 is a 732 million parameter text-to-image latent diffusion model. It is a state-of-the-art diffusion-based text-to-image generation model, builds upon the core architecture of Stable Diffusion. It incorporates key elements like the Variational Autoencoder (VAE) and the pre-trained Text Encoder CLIP. | [Hugging Face](https://huggingface.co/collections/Deci/decidiffusion-models-65a7fc00d0803e7abc1487cc) |
| Playground v2.5 | Playground AI | Playground v2.5 is a diffusion-based text-to-image generative model. It is the state-of-the-art open-source model in aesthetic quality. | [Hugging Face](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) |
| SDXL-Lightning | Bytedance | SDXL-Lightning is a lightning-fast text-to-image generation model. It can generate high-quality 1024px images in a few steps. | [Hugging Face](https://huggingface.co/ByteDance/SDXL-Lightning) |


<br>

### Speech and Audio Models 


|  Model  | Created By | Description |    Link  |
| ------- | ----------- | --------- | ------ | 
| Whisper (STT) | OpenAI | Whisper is a Transformer based encoder-decoder model, also referred to as a sequence-to-sequence model. It was trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper large-v2. | [Hugging Face](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) [Github](https://github.com/openai/whisper) |
| Distil-whisper (STT) | Hugging Face | | [Hugging Face](https://huggingface.co/distil-whisper) [Github](https://github.com/huggingface/distil-whisper) |
| Metavoice (TTS) | MetaVoice | MetaVoice-1B is a 1.2B parameter base model trained on 100K hours of speech for TTS (text-to-speech). | [Hugging Face](https://huggingface.co/metavoiceio/metavoice-1B-v0.1) [Github](https://github.com/metavoiceio/metavoice-src) |
| SpeechT5 (TTS) | Microsoft | SpeechT5 model fine-tuned for speech synthesis (text-to-speech). The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. | [Hugging Face](https://huggingface.co/collections/microsoft/speecht5-650995fc647a3ea442cc6c7b) [Github](https://github.com/microsoft/SpeechT5) |


<br>

---


<br>

## Closed Models 

<br>

### Text Generation 

|  Model  | Created By |  Link  |
| ------- | -----------| ------ |
| GPT4 | OpenAI | [GPT4](https://openai.com/gpt-4) |  
| GPT3.5 | OpenAI |  [GPT3.5](https://platform.openai.com/docs/guides/chat) |  
| Gemini1.5 | Google |  [Gemini](https://deepmind.google/technologies/gemini/) [Blog](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) | 
| Gemini1.0 | Google |  [Gemini](https://deepmind.google/technologies/gemini/) [Blog](https://blog.google/technology/ai/google-gemini-ai/) |  
| Claude3 | Anthropic | 
[Claude](https://www.anthropic.com/claude) [Blog](https://www.anthropic.com/news/claude-3-family) | 
| Claude2.1 | Anthropic |   [Claude](https://www.anthropic.com/claude) [Blog](https://www.anthropic.com/news/claude-2-1) |   
| Mistral Large | Mistral AI |  [Mistral](https://mistral.ai/technology/#models) [Blog](https://mistral.ai/news/mistral-large/) |
| Mistral Medium | Mistral AI |  [Mistral](https://mistral.ai/technology/#models) |
| Palm 2 | Google | [Palm2](https://ai.google/discover/palm2/) |
| Jurassic2 | A121 labs | [Blog](https://www.ai21.com/blog/introducing-j2) |
| Titan | AWS | [Titan](https://aws.amazon.com/bedrock/titan/) |


<br>

### Image Generation

|  Model  | Created By |  Link  |
| ------- | ---------- | ------- | 
| Imagen 2 | Google  |  [Imagen](https://deepmind.google/technologies/imagen-2/) |
| Dalle3 | OpenAI |  [Dalle3](https://openai.com/dall-e-3) |
| Dalle2 | OpenAI |  [Dalle2](https://openai.com/dall-e-2) |
| Firefly | Adobe | [Firefly](https://firefly.adobe.com) | 
| Midjourney v6 | Midjourney | [Midjourney](https://www.midjourney.com/)  |
| Titan Image Generator | AWS | [Titan](https://aws.amazon.com/blogs/aws/amazon-titan-image-generator-multimodal-embeddings-and-text-models-are-now-available-in-amazon-bedrock/) |


<br>

---