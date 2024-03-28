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
| Grok 1.0 | xAI | 314B | Grok 1.0 uses Mixture of 8 Experts (MoE). Grok 1.0 is not fine-tuned for specific applications like dialogue but showcases strong performance compared to other models like GPT-3.5 and Llama 2. It is larger than GPT-3/3.5.  | [Github](https://github.com/xai-org/grok-1) [Hugging Face](https://huggingface.co/xai-org/grok-1) | 
| Llama 2 | Meta AI | 7B 13B 70B | Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. It is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety. | [Hugging Face](https://huggingface.co/meta-llama) [Github](https://github.com/meta-llama/llama) [Blog](https://ai.meta.com/blog/llama-2/) | 
| Dolly v2 | Databricks | 3B 7B 12B | Dolly v2 is a causal language model created by Databricks that is derived from EleutherAI's Pythia-12b and fine-tuned on a ~15K record instruction corpus. | [Hugging Face Dolly 3B](https://huggingface.co/databricks/dolly-v2-3b) [Hugging Face Dolly 7B](https://huggingface.co/databricks/dolly-v2-7b) [Hugging Face Dolly 12B](https://huggingface.co/databricks/dolly-v2-12b) [Github](https://github.com/databrickslabs/dolly) | 
| DBRX | Databricks | 132B Active 36B | DBRX is a transformer-based decoder-only large language model (LLM) that was trained using next-token prediction. It uses a fine-grained mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input. It was pre-trained on 12T tokens of text and code data. Compared to other open MoE models like Mixtral-8x7B and Grok-1, DBRX is fine-grained, meaning it uses a larger number of smaller experts. DBRX has 16 experts and chooses 4, while Mixtral-8x7B and Grok-1 have 8 experts and choose 2. This provides 65x more possible combinations of experts which improves model quality. | [Hugging Face](https://huggingface.co/collections/databricks/dbrx-6601c0852a0cdd3c59f71962) [Github](https://github.com/databricks/dbrx) [Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | 
| Command-R | Cohere | 35B | Command-R is a research release of a 35 billion parameter highly performant generative model. Command-R is a large language model with open weights optimized for a variety of use cases including reasoning, summarization, and question answering. Command-R has the capability for multilingual generation evaluated in 10 languages and highly performant RAG capabilities. | [Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-r-v01) | 
| Qwen1.5 | Alibaba | 0.5B 1.8B 4B 7B 14B 72B | Qwen1.5 is a transformer-based decoder-only language model pretrained on a large amount of data. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, mixture of sliding window attention and full attention, etc. | [Hugging Face](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524) [Github](https://github.com/QwenLM/Qwen) | 
| Vicuna v1.5 | Lysms | 7B 13B  | Vicuna v1.5 is fine-tuned from Llama 2 with supervised instruction fine-tuning. The training data is around 125K conversations collected from ShareGPT.com. The primary use of Vicuna is research on large language models and chatbots. | [Hugging Face Vicuna 7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) [Hugging Face Vicuna 13B](https://huggingface.co/lmsys/vicuna-13b-v1.5) | 
| Phi 2 | Microsoft | 2.7B | Phi-2 is a Transformer with 2.7 billion parameters. It was trained using the same data sources as Phi-1.5, augmented with a new data source that consists of various NLP synthetic texts and filtered websites. When assessed against benchmarks testing common sense, language understanding, and logical reasoning, Phi-2 showcased a nearly state-of-the-art performance among models with less than 13 billion parameters. | [Hugging Face](https://huggingface.co/microsoft/phi-2) |
| Orca 2 | Microsoft | 7B 13B | Orca 2 is built for research purposes only and provides a single turn response in tasks such as reasoning over user given data, reading comprehension, math problem solving and text summarization. The model is designed to excel particularly in reasoning. The model is not optimized for chat and has not been trained with RLHF or DPO. | [Hugging Face](https://huggingface.co/collections/microsoft/orca-65bbeef1980f5719cccc89a3) |
| Smaug | Abacus AI | 34B 72B | Smaug is created using a new fine-tuning technique, DPO-Positive (DPOP), and new pairwise preference versions of ARC, HellaSwag, and MetaMath (as well as other existing datasets). | [Hugging Face](https://huggingface.co/abacusai) | 
| MPT | Mosaicml | 1B 7B 30B | MPT is a decoder-style transformer pretrained from scratch on 1T tokens of English text and code. These models use a modified transformer architecture optimized for efficient training and inference. These architectural changes include performance-optimized layer implementations and the elimination of context length limits by replacing positional embeddings with Attention with Linear Biases (ALiBi). | [Hugging Face](https://huggingface.co/collections/mosaicml/mpt-6564f3d9e5aac326bfa22def) | 
| Falcon | TLL | 7B 40B 180B | Falcon is a 7B/40B/180B parameters causal decoder-only models built by TII and trained on  1,000B/1,500B/3,500B tokens of RefinedWeb enhanced with curated corpora. | [Hugging Face](https://huggingface.co/tiiuae) | 
| DeciLM | DeciAI | 6B 7B | DeciLM is a decoder-only text generation model. With support for an 8K-token sequence length, this highly efficient model uses variable Grouped-Query Attention (GQA) to achieve a superior balance between accuracy and computational efficiency. | [Hugging Face](https://huggingface.co/collections/Deci/decilm-models-65a7fb6a65e4f1a5eb14917a) |
| BERT | Google | 110M to 350M | BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labeling them in any way  with an automatic process to generate inputs and labels from those texts. |
[Hugging Face](https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc) [GitHub](https://github.com/google-research/bert) | 
| Olmo | AllenAI | 1B 7B | OLMo is a series of Open Language Models designed to enable the science of language models. The OLMo models are trained on the Dolma dataset. | [Hugging Face](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) [Github](https://github.com/allenai/OLMo) | 
| Openchat3.5 | Openchat | 7B | Openchat2.5 is the best performing 7B LLM. | [Hugging Face](https://huggingface.co/collections/openchat/openchat-65110500e14eeb01d4888806) [Github](https://github.com/imoneoi/openchat) |
| Bloom | BigScience | | BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text data using industrial-scale computational resources. | [Hugging Face](https://huggingface.co/bigscience/bloom) |
| Hermes 2 Pro Mistral | Nous Research | 7B | Hermes 2 Pro on Mistral 7B is the new flagship 7B Hermes. Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house. This new version of Hermes maintains its excellent general task and conversation capabilities - but also excels at Function Calling, JSON Structured Outputs. | [Hugging Face](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B) | 
| Hermes 2 Mixtral 7x8B | Nous Research |  | Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained over the Mixtral 8x7B MoE LLM. The model was trained on over 1,000,000 entries of primarily GPT-4 generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks. This is the SFT + DPO version of Mixtral Hermes 2. | [Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) |
| Merlinite | IBM | 7B | Merlinite-7b is a Mistral-7b-derivative model trained with the LAB methodology, using Mixtral-8x7b-Instruct as a teacher model. | [Hugging Face](https://huggingface.co/ibm/merlinite-7b) |
| Labradorite | IBM | 13B | Labradorite-13b is a LLaMA-2-13b-derivative model trained with the LAB methodology, using Mixtral-8x7b-Instruct as a teacher model. | [Hugging Face](https://huggingface.co/ibm/labradorite-13b) | 
| Solar | Upstage | 10.7B | SOLAR-10.7B, an advanced large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. It's compact, yet remarkably powerful, and demonstrates unparalleled state-of-the-art performance in models with parameters under 30B. | [Hugging Face](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) |


<br>

### Code Generation

|  Model  | Created By | Size | Description |    Link  |
| ------- | ----------- | --------- | ------ | ----- | 
| Codellama | Meta AI | 7B 13B 34B 70B | Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. | [Hugging Face](https://huggingface.co/codellama) [Github](https://github.com/meta-llama/codellama) | 
| Starcoder | BigCode | 15.5B | The StarCoder models are 15.5B parameter models trained on 80+ programming languages from The Stack (v1.2), with opt-out requests excluded. The model uses Multi Query Attention, a context window of 8192 tokens, and was trained using the Fill-in-the-Middle objective on 1 trillion tokens. | [Hugging Face](https://huggingface.co/bigcode/starcoder) [Github](https://github.com/bigcode-project/starcoder) | 
| Starcoder2 | BigCode | 3B 7B 15B | StarCoder2-15B model is a 15B parameter model trained on 600+ programming languages from The Stack v2, with opt-out requests excluded. The model uses Grouped Query Attention, a context window of 16,384 tokens with a sliding window attention of 4,096 tokens, and was trained using the Fill-in-the-Middle objective on 4+ trillion tokens. | [Hugging Face](https://huggingface.co/collections/bigcode/starcoder2-65de6da6e87db3383572be1a) [GitHub](https://github.com/bigcode-project/starcoder2) | 
| DeciCoder | DeciAI | 1B 6B | DeciCoder are decoder-only code completion models trained on the Python, Java, and Javascript subsets of Starcoder Training Dataset. The model uses Grouped Query Attention and has a context window of 2048 tokens. It was trained using a Fill-in-the-Middle training objective. | [Hugging Face](https://huggingface.co/collections/Deci/decicoder-models-65a7faf617d869bb743f1766) |  
| Stable Code | Stability AI | 3B | stable-code is a 2.7B billion parameter decoder-only language model pre-trained on 1.3 trillion tokens of diverse textual and code datasets. stable-code-3b is trained on 18 programming languages and demonstrates state-of-the-art performance (compared to models of similar size) on the MultiPL-E metrics across multiple programming languages tested using BigCode's Evaluation Harness. | [Hugging Face](https://huggingface.co/collections/stabilityai/stable-code-64f9dfb4ebc8a1be0a3f7650) [Github](https://github.com/Stability-AI/StableCode) | 
| SqlCoder | DefogAI | 7B 15B 34B 70B | Defog's SQLCoder is a state-of-the-art LLM for converting natural language questions to SQL queries. It slightly outperforms gpt-3.5-turbo for natural language to SQL generation tasks on our sql-eval framework, and significantly outperforms all popular open-source models. It also significantly outperforms text-davinci-003, a model that's more than 10 times its size. | [Hugging Face](https://huggingface.co/defog) [Github](https://github.com/defog-ai/sqlcoder) | 
| DeepSeek Coder | Deepseek AI | 1B to 33B | Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. Various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and a extra fill-in-the-blank task, to support project-level code completion and infilling. For coding capabilities, Deepseek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks. | [Hugging Face](https://huggingface.co/collections/deepseek-ai/deepseek-coder-65f295d7d8a0a29fe39b4ec4) [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) | 
| Codegen2 | Salesforce | 1B 3.7B 7B 16B | CodeGen2 is a family of autoregressive language models for program synthesis. CodeGen2 was trained using cross-entropy loss to maximize the likelihood of sequential inputs. The input sequences are formatted in two ways: (1) causal language modeling and (2) file-level span corruption. |  [Hugging Face](https://huggingface.co/Salesforce/codegen2-16B) [Github](https://github.com/salesforce/CodeGen) | 


<br>

### Image Generation

|  Model  | Created By | Description |    Link  |
| ------- | ----------- | --------- | ------ | 
| Stable Diffusion2 | Stability AI | It is a Diffusion-based text-to-image generation model. This model can be used to generate and modify images based on text prompts. It is a Latent Diffusion Model that uses a fixed, pretrained text encoder (OpenCLIP-ViT/H). | [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2) [Github](https://github.com/Stability-AI/stablediffusion) [Github](https://github.com/Stability-AI/stablediffusion) |
| SDXL-turbo | Stability AI | SDXL-Turbo is a fast generative text-to-image model that can synthesize photorealistic images from a text prompt in a single network evaluation. SDXL-Turbo is a distilled version of SDXL 1.0, trained for real-time synthesis. SDXL-Turbo is based on a novel training method called Adversarial Diffusion Distillation (ADD). | [Hugging Face](https://huggingface.co/stabilityai/sdxl-turbo) | 
| Stable Cascade | Stability AI | Stable Cascade is a diffusion model trained to generate images given a text prompt. It  is built upon the Würstchen architecture and its main difference to other models like Stable Diffusion is that it is working at a much smaller latent space. |  [Hugging Face](https://huggingface.co/stabilityai/stable-cascade) [Github](https://github.com/Stability-AI/StableCascade) |
| DeciDiffusion v2.0 | DeciAI | DeciDiffusion 2.0 is a 732 million parameter text-to-image latent diffusion model. It is a state-of-the-art diffusion-based text-to-image generation model, builds upon the core architecture of Stable Diffusion. It incorporates key elements like the Variational Autoencoder (VAE) and the pre-trained Text Encoder CLIP. | [Hugging Face](https://huggingface.co/collections/Deci/decidiffusion-models-65a7fc00d0803e7abc1487cc) |
| Playground v2.5 | Playground AI | Playground v2.5 is a diffusion-based text-to-image generative model. It is the state-of-the-art open-source model in aesthetic quality. | [Hugging Face](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic) |
| SDXL-Lightning | Bytedance | SDXL-Lightning is a lightning-fast text-to-image generation model. It can generate high-quality 1024px images in a few steps. | [Hugging Face](https://huggingface.co/ByteDance/SDXL-Lightning) |
| Open Journey | PromptHero | Stable Diffusion fine tuned model on Midjourney images. | [Hugging Face](https://huggingface.co/prompthero/openjourney) |


<br>

### Speech and Audio Models 


|  Model  | Created By | Description |    Link  |
| ------- | ----------- | --------- | ------ | 
| Whisper (STT) | OpenAI | Whisper is a Transformer based encoder-decoder model, also referred to as a sequence-to-sequence model. It is available in different sizes:- tiny, base, small, medium, large, large-v2, large-v3. It was trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper large-v2. | [Hugging Face](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) [Github](https://github.com/openai/whisper) |
| Distil-whisper (STT) | Hugging Face | Distil Whisper is the knowledge distilled version of OpenAI's Whisper. It is available in different sizes:- distil-small, distil-medium, distil-large-v2, distil-large-v3. | [Hugging Face](https://huggingface.co/distil-whisper) [Github](https://github.com/huggingface/distil-whisper) |
| Metavoice (TTS) | MetaVoice | MetaVoice-1B is a 1.2B parameter base model trained on 100K hours of speech for TTS (text-to-speech). | [Hugging Face](https://huggingface.co/metavoiceio/metavoice-1B-v0.1) [Github](https://github.com/metavoiceio/metavoice-src) |
| SpeechT5 (TTS) | Microsoft | SpeechT5 model fine-tuned for speech synthesis (text-to-speech). The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. | [Hugging Face](https://huggingface.co/microsoft/speecht5_tts) [Github](https://github.com/microsoft/SpeechT5) |
| Magnet | Meta | MAGNeT is a text-to-music and text-to-sound model capable of generating high-quality audio samples conditioned on text descriptions. It is a masked generative non-autoregressive Transformer trained over a 32kHz EnCodec tokenizer with 4 codebooks sampled at 50 Hz. | [GitHub](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MAGNET_MODEL_CARD.md) | 
| Musicgen | Meta | MusicGen is a text-to-music model capable of generating high-quality music samples conditioned on text descriptions or audio prompts. It is a single stage auto-regressive Transformer model trained over a 32kHz EnCodec tokenizer with 4 codebooks sampled at 50 Hz. Various Sizes of Musicgen:- musicgen-small, musicgen-medium, musicgen-melody, musicgen-large, musicgen-melody-large, musicgen-stereo | [Hugging Face](https://huggingface.co/collections/facebook/musicgen-stereo-654bcd4509dd7ef5247f3bdf) [Github](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md) | 



<br>

### Video Generation Models 


|  Model  | Created By | Description |    Link  |
| ------- | ----------- | --------- | ------ | 
| Stable Video Diffusion | Stability AI | Stable Video Diffusion (SVD) Image-to-Video is a diffusion model that takes in a still image as a conditioning frame, and generates a video from it. It is a latent diffusion model trained to generate short video clips from an image conditioning. This model was trained to generate 14 frames at resolution 576x1024 given a context frame of the same size. | [Hugging Face](https://huggingface.co/collections/stabilityai/video-65f87e5fc8f264ce4dae9bfa) |
| AnimateDiff Lightening | Bytedance | AnimateDiff-Lightning is a lightning-fast text-to-video generation model. It can generate videos more than ten times faster than the original AnimateDiff. AnimateDiff-Lightning produces the best results when used with stylized base models. | [Hugging Face](https://huggingface.co/ByteDance/AnimateDiff-Lightning) | 


<br>

---


<br>

## Closed Source (Proprietary) Models 

<br>

### Text Generation 

|  Model  | Created By |  Link  |
| ------- | -----------| ------ |
| GPT4 | OpenAI | [GPT4](https://openai.com/gpt-4) |  
| GPT3.5 | OpenAI |  [GPT3.5](https://platform.openai.com/docs/guides/chat) |  
| Gemini1.5 | Google |  [Gemini](https://deepmind.google/technologies/gemini/) [Blog](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) | 
| Gemini1.0 | Google |  [Gemini](https://deepmind.google/technologies/gemini/) [Blog](https://blog.google/technology/ai/google-gemini-ai/) |  
| Claude3 | Anthropic | [Claude](https://www.anthropic.com/claude) [Blog](https://www.anthropic.com/news/claude-3-family) | 
| Claude2.1 | Anthropic |   [Claude](https://www.anthropic.com/claude) [Blog](https://www.anthropic.com/news/claude-2-1) |   
| Mistral Large | Mistral AI |  [Mistral](https://mistral.ai/technology/#models) [Blog](https://mistral.ai/news/mistral-large/) |
| Mistral Medium | Mistral AI |  [Mistral](https://mistral.ai/technology/#models) |
| Palm 2 | Google | [Palm2](https://ai.google/discover/palm2/) |
| Jurassic2 | A121 labs | [Blog](https://www.ai21.com/blog/introducing-j2) |
| Titan | AWS | [Titan](https://aws.amazon.com/bedrock/titan/) |
| Granite | IBM | [Granite](https://www.ibm.com/products/watsonx-ai/foundation-models) |


<br>

### Image Generation

|  Model  | Created By |  Link  |
| ------- | ---------- | ------- | 
| Imagen 2 | Google  |  [Imagen](https://deepmind.google/technologies/imagen-2/) |
| Dalle3 | OpenAI |  [Dalle3](https://openai.com/dall-e-3) |
| Dalle2 | OpenAI |  [Dalle2](https://openai.com/dall-e-2) |
| Firefly | Adobe | [Firefly](https://firefly.adobe.com) | 
| Midjourney v6, v5 | Midjourney | [Midjourney](https://www.midjourney.com/)  |
| Titan Image Generator | AWS | [Titan](https://aws.amazon.com/blogs/aws/amazon-titan-image-generator-multimodal-embeddings-and-text-models-are-now-available-in-amazon-bedrock/) |
| Emu Edit | Meta | [Emu Edit](https://emu-edit.metademolab.com/) [Blog](https://ai.meta.com/blog/emu-text-to-video-generation-image-editing-research/) | 


<br>

### Video Generation Models 


|  Model  | Created By |  Link  |
| ------- | ----------- | ------ | 
| Sora | OpenAI | [Sora](https://openai.com/sora) |
| Runwayml Gen2 | Runwayml | [Gen2](https://research.runwayml.com/gen2) | 
| Runwayml Gen1 | Runwayml | [Gen1](https://research.runwayml.com/gen1) |
| Emu Video | Meta | [Emu Video](https://emu-video.metademolab.com/) [Blog](https://ai.meta.com/blog/emu-text-to-video-generation-image-editing-research/) | 

---