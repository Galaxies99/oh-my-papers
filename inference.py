import os
import yaml
import json
import argparse
import logging
from utils.logger import ColoredLogger
from inference_bert import BertInferencer
from inference_citation_bert import CitationBertInferencer
from inference_vgae import VGAEInferencer


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class Inferencer(object):
    def __init__(self, engine_cfg, relation_engine_cfg, engine = "citation_bert", relation_engine = "VGAE"):
        if engine not in ["citation_bert", "bert"] or relation_engine not in ["VGAE"]:
            raise AttributeError('Invalid engine or invalid relation engine: engine should be either "citation_bert" or "bert" and relation engine should be "VGAE"')
        if engine == "citation_bert":
            self.engine = CitationBertInferencer(**engine_cfg)
        elif engine == "bert":
            self.engine = BertInferencer(**engine_cfg)
        if relation_engine == "VGAE":
            self.relation_engine = VGAEInferencer(**relation_engine_cfg)
        
    def context_inference(self, input_dict):
        logger.info('Begin context inference ...')
        res = self.engine.inference(input_dict)
        logger.info('Context inference finished.')
        return res
    
    def citation_inference(self, citation_dict, placeholder = '[?]'):
        logger.info('Begin citation inference ...')
        context = citation_dict["context"]
        contexts = []
        pos = context.find(placeholder)
        while pos != -1:
            contexts.append(context[:pos])
            context = context[pos + len(placeholder):]
            pos = context.find(placeholder)
        contexts.append(context)
        input_dict = {'inference': []}
        for i in range(len(contexts) - 1):
            input_dict['inference'].append({
                'left_context': contexts[i],
                'right_context': contexts[i + 1]
            })
        if input_dict['inference'] == []:
            return {"context": context, "references": []}
        res = self.engine.inference(input_dict)
        res = res['inference']
        new_context = ""
        references = []
        for i, context in enumerate(contexts):
            new_context = new_context + context
            if i == len(contexts) - 1:
                break
            ref_info = res[i]['result'][0]
            references.append(ref_info)
            new_context = new_context + "[" + str(i + 1) + "]"
        logger.info('Citation inference finished.')
        return {"context": new_context, "references": references}
    
    def relation_inference(self, node_info, k = 10):
        logger.info('Begin relation inference ...')
        res = self.relation_engine.find_topk(node_info, k = k)
        logger.info('Relation inference finished.')
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_cfg', default = os.path.join('configs', 'citation_bert.yaml'), help = 'Configuration file of the main engine', type = str)
    parser.add_argument('--relation_engine_cfg', default = os.path.join('configs', 'vgae.yaml'), help = 'Configuration file of the relationship engine', type = str)
    parser.add_argument('--context_input', default = os.path.join('examples', 'context.json'), help = 'input file of the context inference', type = str)
    parser.add_argument('--context_output', default = os.path.join('examples', 'context-res.json'), help = 'output file of the context inference', type = str)
    parser.add_argument('--citation_input', default = os.path.join('examples', 'citation.json'), help = 'input file of the citation inference', type = str)
    parser.add_argument('--citation_output', default = os.path.join('examples', 'citation-res.json'), help = 'output file of the citation inference', type = str)
    parser.add_argument('--relation_input', default = os.path.join('examples', 'relation.json'), help = 'input file of the relation inference', type = str)
    parser.add_argument('--relation_output', default = os.path.join('examples', 'relation-res.json'), help = 'output file of the relation inference', type = str)

    FLAGS = parser.parse_args()
    ENGINE_CFG_FILE = FLAGS.engine_cfg
    RELATION_ENGINE_CFG_FILE = FLAGS.relation_engine_cfg
    CONTEXT_INPUT_FILE = FLAGS.context_input
    CONTEXT_OUTPUT_FILE = FLAGS.context_output
    CITATION_INPUT_FILE = FLAGS.citation_input
    CITATION_OUTPUT_FILE = FLAGS.citation_output
    RELATION_INPUT_FILE = FLAGS.relation_input
    RELATION_OUTPUT_FILE = FLAGS.relation_output

    with open(ENGINE_CFG_FILE, 'r') as cfg_file:
        engine_cfg = yaml.load(cfg_file, Loader = yaml.FullLoader)
    with open(RELATION_ENGINE_CFG_FILE, 'r') as cfg_file:
        relation_engine_cfg = yaml.load(cfg_file, Loader = yaml.FullLoader)

    if os.path.exists(os.path.dirname(CONTEXT_OUTPUT_FILE)) == False:
        os.makedirs(os.path.dirname(CONTEXT_OUTPUT_FILE))
    if os.path.exists(os.path.dirname(CITATION_OUTPUT_FILE)) == False:
        os.makedirs(os.path.dirname(CITATION_OUTPUT_FILE))
    if os.path.exists(os.path.dirname(RELATION_OUTPUT_FILE)) == False:
        os.makedirs(os.path.dirname(RELATION_OUTPUT_FILE))

    inferencer = Inferencer(engine_cfg, relation_engine_cfg, engine = "citation_bert", relation_engine = "VGAE")

    with open(CONTEXT_INPUT_FILE, 'r') as f:
        input_dict = json.load(f)
    output_dict = inferencer.context_inference(input_dict)
    with open(CONTEXT_OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f)
    
    with open(CITATION_INPUT_FILE, 'r') as f:
        input_dict = json.load(f)
    output_dict = inferencer.citation_inference(input_dict)
    with open(CITATION_OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f)
    
    with open(RELATION_INPUT_FILE, 'r') as f:
        input_dict = json.load(f)
    output_dict = inferencer.relation_inference(input_dict)
    with open(RELATION_OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f)