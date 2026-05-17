from dataclasses import dataclass
import json
import sys
import types
import unittest


class FakeNumpyGeneric:
    def item(self):
        return float(self)


class FakeFloat32(float, FakeNumpyGeneric):
    pass


class FakeNDArray:
    def __class_getitem__(cls, item):
        return cls


fake_numpy = types.ModuleType("numpy")
fake_numpy.generic = FakeNumpyGeneric
fake_numpy.float32 = FakeFloat32
fake_numpy_typing = types.ModuleType("numpy.typing")
fake_numpy_typing.NDArray = FakeNDArray
sys.modules.setdefault("numpy", fake_numpy)
sys.modules.setdefault("numpy.typing", fake_numpy_typing)

from utils import to_rerank_response


@dataclass
class DataclassRerankItem:
    relevance_score: float
    document: str
    index: int


class AttributeRerankItem:
    def __init__(self, score, index):
        self.score = score
        self.index = index


class RerankResponseTests(unittest.TestCase):
    def test_float_scores_are_json_serializable(self):
        response = to_rerank_response(
            scores=[0.9, FakeFloat32(0.1)],
            model="reranker",
            usage=12,
        )

        json.dumps(response)
        self.assertEqual(
            response["results"],
            [
                {"relevance_score": 0.9, "index": 0},
                {"relevance_score": float(FakeFloat32(0.1)), "index": 1},
            ],
        )

    def test_dataclass_rerank_items_are_json_serializable(self):
        response = to_rerank_response(
            scores=[
                DataclassRerankItem(
                    relevance_score=0.95,
                    document="matched document",
                    index=2,
                )
            ],
            documents=["a", "b", "fallback document"],
            model="reranker",
            usage=7,
        )

        json.dumps(response)
        self.assertEqual(
            response["results"],
            [
                {
                    "relevance_score": 0.95,
                    "index": 2,
                    "document": "matched document",
                }
            ],
        )

    def test_attribute_rerank_items_can_map_documents_by_index(self):
        response = to_rerank_response(
            scores=[AttributeRerankItem(score=FakeFloat32(0.8), index=1)],
            documents=["wrong", "right"],
            model="reranker",
            usage=5,
        )

        json.dumps(response)
        self.assertEqual(response["results"][0]["document"], "right")
        self.assertEqual(response["results"][0]["index"], 1)
        self.assertEqual(response["results"][0]["relevance_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
