"""
Testes unitários da paginação genérica (src/historical_dataset/paginator.py).

Cobre: extração de itens de array simples vs. `{"results": [...]}`,
avanço de offset, paragem no fim dos dados (página incompleta ou vazia),
e o limite opcional `max_pages`.
"""

import unittest

from src.historical_dataset.paginator import extract_items, iter_endpoint


class FakeClient:
    """Cliente falso que devolve páginas pré-definidas por (endpoint, offset)."""

    def __init__(self, pages):
        self.pages = pages  # dict[int offset] -> payload
        self.calls = []

    def get(self, endpoint, params=None):
        self.calls.append((endpoint, dict(params or {})))
        offset = (params or {}).get("offset", 0)
        return self.pages.get(offset, [])


class TestExtractItems(unittest.TestCase):

    def test_none_returns_empty_list(self):
        self.assertEqual(extract_items(None), [])

    def test_plain_list_returned_as_is(self):
        self.assertEqual(extract_items([1, 2, 3]), [1, 2, 3])

    def test_results_wrapper_extracted(self):
        self.assertEqual(extract_items({"results": [1, 2]}), [1, 2])

    def test_dict_without_results_returns_empty(self):
        self.assertEqual(extract_items({"count": 0}), [])

    def test_other_types_return_empty(self):
        self.assertEqual(extract_items("not a payload"), [])


class TestIterEndpoint(unittest.TestCase):

    def test_stops_when_page_shorter_than_page_size(self):
        client = FakeClient({0: [{"id": 1}, {"id": 2}]})

        items = list(iter_endpoint(client, "events/", page_size=5))

        self.assertEqual([i["id"] for i in items], [1, 2])
        self.assertEqual(len(client.calls), 1)

    def test_advances_offset_across_full_pages(self):
        client = FakeClient({
            0: [{"id": 1}, {"id": 2}],
            2: [{"id": 3}, {"id": 4}],
            4: [{"id": 5}],
        })

        items = list(iter_endpoint(client, "events/", page_size=2))

        self.assertEqual([i["id"] for i in items], [1, 2, 3, 4, 5])
        self.assertEqual(len(client.calls), 3)

    def test_empty_page_stops_iteration(self):
        client = FakeClient({0: [{"id": 1}, {"id": 2}], 2: []})

        items = list(iter_endpoint(client, "events/", page_size=2))

        self.assertEqual([i["id"] for i in items], [1, 2])

    def test_results_wrapper_pages_supported(self):
        client = FakeClient({
            0: {"results": [{"id": 1}, {"id": 2}]},
            2: {"results": []},
        })

        items = list(iter_endpoint(client, "events/", page_size=2))

        self.assertEqual([i["id"] for i in items], [1, 2])

    def test_max_pages_limits_requests(self):
        client = FakeClient({
            0: [{"id": 1}, {"id": 2}],
            2: [{"id": 3}, {"id": 4}],
            4: [{"id": 5}, {"id": 6}],
        })

        items = list(iter_endpoint(client, "events/", page_size=2, max_pages=2))

        self.assertEqual([i["id"] for i in items], [1, 2, 3, 4])
        self.assertEqual(len(client.calls), 2)

    def test_base_params_forwarded_to_every_page(self):
        client = FakeClient({0: [{"id": 1}]})

        list(iter_endpoint(client, "events/", params={"league_id": 39, "status": "finished"}, page_size=100))

        endpoint, params = client.calls[0]
        self.assertEqual(endpoint, "events/")
        self.assertEqual(params["league_id"], 39)
        self.assertEqual(params["status"], "finished")
        self.assertEqual(params["limit"], 100)
        self.assertEqual(params["offset"], 0)


if __name__ == "__main__":
    unittest.main()
