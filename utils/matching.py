from fuzzywuzzy import process

class FuzzyMatchTool:
    def __init__(self, node_info):
        self.paper_info = node_info
        self.num_papers = len(node_info)
        self.title_list = [paper['title'] for paper in node_info]
        self.title2id = dict(zip(self.title_list, range(self.num_papers)))

    def match(self, sentence, topk=5, threshold=80):
        topk = int(topk)
        assert topk <= self.num_papers and threshold <= 100 and threshold > 0

        candidates = process.extract(sentence, self.title_list, limit=topk)
        result = [[self.title2id[title], score, self.paper_info[self.title2id[title]]] for (title, score) in candidates if score > threshold]

        return result

if __name__ == '__main__':
    paper_info = [
        {'title': 'Resnet in Resnet: Generalizing Residual Architectures', 'x': 'test'},
        {'title': 'Wider or Deeper: Revisiting the ResNet Model for Visual Recognition', 'x': 'test'},
        {'title': 'Inception-v4, inception-resnet and the impact of residual connections on learning', 'x': 'test'},
        {'title': 'Demystifying resnet', 'x': 'test'}
    ]

    fuzzy_match_tool = FuzzyMatchTool(paper_info)
    print(fuzzy_match_tool.match('resnet', topk=4, threshold=10))