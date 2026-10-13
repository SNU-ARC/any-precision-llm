import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from any_precision.quantization.gradients import get_gradients


class CalibrationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 1, bias=False, device='cuda', dtype=torch.float16)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[1.001, 0.5]], device='cuda', dtype=torch.float16))

    @property
    def device(self):
        return self.proj.weight.device

    def forward(self, input_ids, labels):
        return SimpleNamespace(loss=self.proj(input_ids.to(self.proj.weight.dtype)).square().mean())


class GradientStateDictTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA to check retained GPU tensors')
    def test_original_weights_are_preserved_on_cpu(self):
        model = CalibrationModel()
        analyzer = SimpleNamespace(
            model=model, tokenizer=None, state_dict=model.state_dict(),
            get_layers=lambda: [model], get_modules=lambda layer: {'proj': layer.proj},
        )
        original = analyzer.state_dict['proj.weight'].cpu().clone()
        tokens = torch.tensor([1, 2], dtype=torch.int64)
        with patch('any_precision.quantization.gradients.get_tokens', return_value=[tokens]):
            gradients = get_gradients(analyzer, dataset='c4', seq_len=2, num_examples=1, random_state=42)

        saved = analyzer.state_dict['proj.weight']
        self.assertEqual(saved.device.type, 'cpu')
        self.assertEqual(saved.dtype, torch.float16)
        torch.testing.assert_close(saved, original, rtol=0, atol=0)
        # The original FP16 value must survive calibration's BF16 rounding.
        self.assertFalse(torch.equal(saved, model.proj.weight.float()))
        self.assertEqual(model.proj.weight.dtype, torch.bfloat16)

        reference = original.to(torch.bfloat16).requires_grad_()
        loss = (tokens.to(torch.bfloat16).unsqueeze(0) @ reference.T).square().mean()
        expected = torch.autograd.grad(loss, reference)[0].square()
        torch.testing.assert_close(gradients[0]['proj'], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
