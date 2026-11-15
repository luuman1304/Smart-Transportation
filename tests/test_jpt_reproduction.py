import copy
import json
import tempfile
import unittest
from pathlib import Path

from lastmile_data.reproduce_jpt import load_scorer, official_score, read_tour


class JPTTests(unittest.TestCase):
    def test_official_scorer_and_cost_preservation(self):
        nodes = ['AA', 'BB', 'CC', 'DD']
        costs = {a: {b: 0 if a == b else (i+1)*(j+3) for j,b in enumerate(nodes)}
                 for i,a in enumerate(nodes)}
        original = copy.deepcopy(costs)
        self.assertEqual(official_score(nodes,nodes,costs),0)
        proposal = ['AA','CC','BB','DD']
        expected = load_scorer().score(nodes+nodes[:1],proposal+proposal[:1],copy.deepcopy(costs))
        self.assertAlmostEqual(official_score(nodes,proposal,costs),expected,places=14)
        self.assertGreater(expected,0)
        self.assertEqual(costs,original)
        with self.assertRaises(ValueError):
            official_score(nodes,['AA','BB','BB','DD'],costs)

    def test_official_invalid_submission_penalty(self):
        nodes = ['AA','BB','CC']
        costs = {a:{b:float(i+j+1) if a!=b else 0.0 for j,b in enumerate(nodes)} for i,a in enumerate(nodes)}
        rid='RouteID_test'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contents=[{rid:{'actual':dict(zip(nodes,range(3)))}}, {}, {rid:costs}, {rid:7.5}]
            paths=[]
            for i,value in enumerate(contents):
                path=root/f'{i}.json'; path.write_text(json.dumps(value)); paths.append(str(path))
            result=load_scorer().evaluate(*paths)
            self.assertEqual(result['route_scores'][rid],7.5)
            self.assertFalse(result['route_feasibility'][rid])

    def test_tour_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tour'
            path.write_text('NAME : amz0001\nCOMMENT : Cost = 2_12345\nTOUR_SECTION\n3\n1\n2\n-1\nEOF\n')
            self.assertEqual(read_tour(path),(12345,['3','1','2']))
