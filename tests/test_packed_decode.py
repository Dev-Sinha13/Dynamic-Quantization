import unittest

try:
    import torch
    from anchorkv.packed_decode import PagedLayer, choose_pages, dense_attention, pack_page
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch dependency is unavailable')
class PagedDecodeTests(unittest.TestCase):
    def test_quantization_counts_storage_and_handles_tiny_values(self):
        torch.manual_seed(11)
        source = torch.randn(2, 16, 128).half()
        sizes = []
        for bits in (4, 8, 16):
            packed = pack_page(source, bits)
            sizes.append(packed.nbytes)
            self.assertEqual(packed.dense().shape, source.shape)
            self.assertTrue(torch.isfinite(packed.dense()).all())
            expected = source.numel() * bits // 8
            if bits != 16:
                expected += source.numel() // 64 * 2
            self.assertEqual(packed.nbytes, expected)
            if bits == 16:
                self.assertNotEqual(source.data_ptr(), packed.data.data_ptr())
                torch.testing.assert_close(packed.dense(), source, atol=0, rtol=0)
        self.assertLess(sizes[0], sizes[1])
        self.assertLess(sizes[1], sizes[2])
        tiny = torch.full_like(source, 1e-7)
        self.assertTrue(torch.isfinite(pack_page(tiny, 4).dense()).all())

    def test_append_demotion_protection_and_partial_page(self):
        layer = PagedLayer(2, 128, 128, protected=(0,), recent_pages=1, device='cpu')
        source = torch.randn(1, 2, 83, 128).half()
        for start in range(0, 83, 7):
            layer.append(source[:, :, start:start + 7], source[:, :, start:start + 7])
        self.assertEqual(layer.length, 83)
        self.assertEqual([pair[0].bits for pair in layer.pages], [16, 4, 4, 4, 16, 16])
        self.assertEqual(layer.demotions, 3)
        k, v = layer.dense()
        self.assertEqual(k.shape, source.shape)
        torch.testing.assert_close(k[:, :, :16], source[:, :, :16], atol=0, rtol=0)
        torch.testing.assert_close(k[:, :, 64:], source[:, :, 64:], atol=0, rtol=0)
        torch.testing.assert_close(k, v)
        with self.assertRaises(ValueError):
            layer.protect(1)
        with self.assertRaises(ValueError):
            layer.demote(0, 4)
        with self.assertRaises(ValueError):
            layer.demote(1, 16)

    def test_equal_budget_selectors_have_identical_allocations(self):
        source = torch.randn(1, 2, 129, 128).half()
        candidates = range(1, 6)
        scores = {page: float(page % 3) for page in candidates}
        sizes = []
        for name in ('random', 'recent', 'automatic', 'oracle'):
            chosen = choose_pages(name, candidates, 2, scores=scores, evidence=(2, 3))
            self.assertEqual(len(chosen), 2)
            layer = PagedLayer(2, 128, 160, protected=chosen | {0}, recent_pages=2, device='cpu')
            layer.append(source, source)
            sizes.append(layer.resident_bytes)
        self.assertEqual(len(set(sizes)), 1)
        self.assertEqual(choose_pages('oracle', candidates, 2, evidence=(2, 3)), {2, 3})

    def test_fp16_attention_matches_original_tensors_with_gqa(self):
        torch.manual_seed(2)
        key = torch.randn(1, 2, 37, 64).half()
        value = torch.randn_like(key)
        query = torch.randn(1, 4, 1, 64).half()
        layer = PagedLayer(2, 64, 64, archive_bits=16, device='cpu')
        layer.append(key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1), scale=0.125,
        )
        torch.testing.assert_close(dense_attention(query, layer, 0.125), expected)

    def test_online_anchor_prevents_later_demotion(self):
        layer = PagedLayer(1, 64, 96, recent_pages=1, device='cpu')
        values = torch.ones(1, 1, 16, 64).half()
        layer.append(values, values)
        layer.protect(0)
        for _ in range(4):
            layer.append(values, values)
        self.assertEqual(layer.pages[0][0].bits, 16)
        self.assertEqual(layer.pages[1][0].bits, 4)


if __name__ == '__main__':
    unittest.main()
