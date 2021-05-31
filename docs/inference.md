# Inference

We construct three modules for inference along with a combined inference module. You can call the inference modules conveniently.

To start inference stage, you need to have pretrained models or your own training models stored in the path specified in the configuration files.

## VGAE (Model 1) Inference

### Basic Settings

The VGAE inference module is implemented in `inference_vgae.py` with the name of `VGAEInferencer`. The simplest way to call the module is calling the program with input and output files detached. Specifically, use the following commands to call the inference module of VGAE model:

```bash
python inference_vgae.py --cfg [Configuration File] --input [Input File] --output [Output File]
```

where `[Configuration File]` is the path to configuration file of the VGAE model (default: `configs/vgae.yaml`), `[Input File]` is the path to input file of VGAE inference (default: `examples/relation.json`), and `[Output File]` is the path to output file of VGAE inference (default: `examples/relation-res.json`).

The input should be a json file with the format of

```json
{
    "id": xxx
}
```

which specifies the id of the paper in the paper dataset. And the output is also a json file with the format of

```json
{
    "result":
    [
        {
            "title": xxx,
            "abstract": xxx,
            "venue": xxx,
            "year": xxx,
            "id": xxx
        },
        ...
    ]
}
```

### Advanced Settings

You can initialize the `VGAEInferencer` using your own configurations. You can just pack the configurations in the configuration file into a dict, and sent it to the module as initilization configurations. It may take some time to initialize the models.

Then, you can call `VGAEInferencer.find_topk(input_dict, k)` to get the result. The `input_dict` should have the same format as the input json file mentioned before, and `k` is the optional integer specifying how many related results you want to fetch (default: 10). Then, the function will return a output dict with the same format as the output json file mentioned before. The inference stage is almost real-time.

Here is an example:

```python
import json
import yaml
from inference_vgae import VGAEInferencer

with open('configs/vgae.yaml', 'r') as cfg_file:
    cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)

inferencer = VGAEInferencer(**cfgs)

with open('examples/relation.json', 'r') as f:
    input_dict = json.load(f)

output_dict = inferencer.find_topk(input_dict)

print(output_dict)
```

## Bert (Model 2) Inference

### Basic Settings

The Bert inference module is implemented in `inference_bert.py` with the name of `BertInferencer`. The simplest way to call the module is calling the program with input and output files detached. Specifically, use the following commands to call the inference module of Bert model:

```bash
python inference_bert.py --cfg [Configuration File] --input [Input File] --output [Output File]
```

where `[Configuration File]` is the path to configuration file of the bert model (default: `configs/bert.yaml`), `[Input File]` is the path to input file of bert inference (default: `examples/context.json`), and `[Output File]` is the path to output file of bert inference (default: `examples/context-res.json`).

The input should be a json file with the format of

```json
{
    "inference": [
        {
            "context": xxx
        },
        {
            "left_context": xxx,
            "right_context": xxx
        },
        ...
    ]
}
```

which specifies the citation context (or left context and right context). And the output is also a json file with the format of

```json
{
    "inference": [
        {
            "result": [
                {
                    "title": xxx,
                    "abstract": xxx,
                    "venue": xxx,
                    "year": xxx,
                    "id": xxx
                },
                ...
            ],
        },
        ...
    ]
}
```

### Advanced Settings

You can initialize the `BertInferencer` using your own configurations. You can just pack the configurations in the configuration file into a dict, and sent it to the module as initilization configurations. It may take some time to initialize the models.

Then, you can call `BertInferencer.inference(input_dict)` to get the result. The `input_dict` should have the same format as the input json file mentioned before. Then, the function will return a output dict with the same format as the output json file mentioned before. The inference stage is almost real-time.

Here is an example:

```python
import json
import yaml
from inference_bert import BertInferencer

with open('configs/bert.yaml', 'r') as cfg_file:
    cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)

inferencer = BertInferencer(**cfgs)

with open('examples/context.json', 'r') as f:
    input_dict = json.load(f)

output_dict = inferencer.inference(input_dict)

print(output_dict)
```

## Citation-bert (Model 3) Inference

### Basic Settings

The Citation-bert inference module is implemented in `inference_citation_bert.py` with the name of `CitationBertInferencer`. The simplest way to call the module is calling the program with input and output files detached. Specifically, use the following commands to call the inference module of Citation-bert model:

```bash
python inference_citation_bert.py --cfg [Configuration File] --input [Input File] --output [Output File]
```

where `[Configuration File]` is the path to configuration file of the citation-bert model (default: `configs/citation_bert.yaml`), `[Input File]` is the path to input file of citatino-bert inference (default: `examples/context.json`), and `[Output File]` is the path to output file of citation-bert inference (default: `examples/context-res.json`).

The input should be a json file with the format specified in the bert inference section, and the output has the format specified in the bert inference section.

### Advanced Settings

You can initialize the `CitationBertInferencer` using your own configurations. You can just pack the configurations in the configuration file into a dict, and sent it to the module as initilization configurations. It may take some time to initialize the models.

Then, you can call `CitationBertInferencer.inference(input_dict)` to get the result. The `input_dict` should have the same format as the input json file mentioned before. Then, the function will return a output dict with the same format as the output json file mentioned before. The inference stage is almost real-time.

Here is an example:

```python
import json
import yaml
from inference_citation_bert import CitationBertInferencer

with open('configs/citation_bert.yaml', 'r') as cfg_file:
    cfgs = yaml.load(cfg_file, Loader = yaml.FullLoader)

inferencer = CitationBertInferencer(**cfgs)

with open('examples/context.json', 'r') as f:
    input_dict = json.load(f)

output_dict = inferencer.inference(input_dict)

print(output_dict)
```

## Combined Inference

### Basic Settings

This inference combined three models together, and add a new function: auto-citation. The combined inference module is implemented in `inference.py` with the name of `Inferencer`. The simplest way to call the module is calling the program with input and output files detached. Specifically, use the following commands to call the inference module:

```bash
python inference.py --engine_cfg [Engine Configuration File]
                    --relation_engine_cfg [Relation Engine Configuration File]
                    --context_input [Context Input File]
                    --context_output [Context Output File]
                    --citation_input [Citation Input File]
                    --citation_output [Citation Output File]
                    --relation_input [Relation Input File]
                    --relation_output [Relation Output File]
```

where 
- `[Engine Configuration File]` is the path to configuration file of the engine (default: citation-bert model, `configs/citation_bert.yaml`);
- `[Relation Engine Configuration File]` is the path to the configuration file of the relation engine (default: VGAE model, `configs/vgae.yaml`);
-  `[Context Input File]` is the path to input file of context inference (default: `examples/context.json`), and `[Context Output File]` is the path to output file of context inference (default: `examples/context-res.json`);
-  `[Citation Input File]` is the path to input file of citation inference (default: `examples/citation.json`), and `[Citation Output File]` is the path to output file of citation inference (default: `examples/citation-res.json`);
-  `[Relation Input File]` is the path to input file of relation inference (default: `examples/relation.json`), and `[Relation Output File]` is the path to output file of relation inference (default: `examples/relation-res.json`).

The format of input and output of context inference and relation inference have been introduced in previous sections respectively. Now let us introduce the input and output format of citation context. The input should be a json file with the format of

```json
{
    "context": xxx
}
```

which specifies the input context to the citation inference with the placeholder of `[?]`. And the output is also a json file with the format of

```json
{
    "context": xxx,
    "references": [
        {
            "title": xxx,
            "abstract": xxx,
            "venue": xxx,
            "year": xxx,
            "id", xxx
        },
        ...
    ]
}
```

The `context` part in the output file replaces placeholder `[?]` with the refernces `[1], [2], ...` and provide the reference papers in the `references` part in the corresponding order.

### Advanced Settings

You can initialize the `Inferencer` using your own configurations. You can set the main engine and relation engine. The main engine should be one of `{"bert", "citation_bert"}`, and the relation engine should be `"VGAE"`. You can just pack the configurations in the configuration files into dicts, and sent them to the module as initilization configurations of each engine. It may take some time to initialize the models.

Then, you can call `Inferencer.context_inference(input_dict)`, `Inferencer.citation_inference(input_dict)` and `Inferencer.relation_inference(input_dict, k)` to get the result of context inference, citation inference and relation inference respectively. The `input_dict` should have the same format as the input json file mentioned before. Then, the function will return a output dict with the same format as the output json file mentioned before. The inference stage is almost real-time.

Here is an example:

```python
import json
import yaml
from inference import Inferencer

with open('configs/citation_bert.yaml', 'r') as cfg_file:
    engine_cfg = yaml.load(cfg_file, Loader = yaml.FullLoader)
with open('configs/vgae.yaml', 'r') as cfg_file:
    relation_engine_cfg = yaml.load(cfg_file, Loader = yaml.FullLoader)

inferencer = Inferencer(engine_cfg, relation_engine_cfg, engine = "citation_bert", relation_engine = "VGAE")

with open('examples/context.json', 'r') as f:
    input_dict = json.load(f)
output_dict = inferencer.context_inference(input_dict)
print(output_dict)
    
with open('examples/citation.json', 'r') as f:
    input_dict = json.load(f)
output_dict = inferencer.citation_inference(input_dict)
print(output_dict)
    
with open('examples/relation.json', 'r') as f:
    input_dict = json.load(f)
output_dict = inferencer.relation_inference(input_dict)
print(output_dict)
```
