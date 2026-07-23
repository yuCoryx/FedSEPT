import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from opacus.accountants import RDPAccountant

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import noise_backend


def _trainer(batch_size, clip=1.0, microbatch_size=2):
    return SimpleNamespace(
        args=SimpleNamespace(
            dp_microbatch_size=microbatch_size,
            dp_delta=1e-5,
        ),
        dp_clip=clip,
        dp_sigma=1e-12,
        _dp_current_client=0,
        _dp_local_sizes={0: batch_size},
        _dp_accountants={0: RDPAccountant()},
        _dp_step_count_by_client={0: 0},
    )


def _reference_clipped_mean(losses, parameters, clip):
    samples = [[] for _ in parameters]
    for index in range(losses.numel()):
        gradients = torch.autograd.grad(
            losses[index],
            parameters,
            retain_graph=True,
        )
        norm = torch.sqrt(
            sum(gradient.float().square().sum() for gradient in gradients)
        )
        factor = min(1.0, clip / max(norm.item(), 1e-12))
        for bucket, gradient in zip(samples, gradients):
            bucket.append(gradient.detach() * factor)
    return [torch.stack(bucket).mean(dim=0) for bucket in samples]


class PerSampleDPSGDTest(unittest.TestCase):
    def test_chunked_joint_clipping_matches_sample_loop(self):
        weight = torch.nn.Parameter(
            torch.tensor([[0.2, -0.3], [0.4, 0.1]])
        )
        bias = torch.nn.Parameter(torch.tensor([0.05, -0.02]))
        inputs = torch.tensor(
            [[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75], [2.0, -1.0]]
        )
        targets = torch.tensor(
            [[0.2, -0.1], [0.5, 0.3], [-0.2, 0.7], [1.0, -0.5]]
        )
        predictions = inputs @ weight.t() + bias
        losses = (predictions - targets).square().mean(dim=1)
        expected = _reference_clipped_mean(
            losses,
            [weight, bias],
            clip=0.7,
        )

        trainer = _trainer(losses.numel(), clip=0.7)
        torch.manual_seed(7)
        updates = noise_backend.compute_private_gradients(
            trainer=trainer,
            per_sample_losses=losses,
            params=[weight, bias],
        )
        losses.mean().backward()
        noise_backend.apply_private_gradients(updates)

        self.assertTrue(torch.allclose(weight.grad, expected[0], atol=1e-6))
        self.assertTrue(torch.allclose(bias.grad, expected[1], atol=1e-6))
        self.assertEqual(trainer._dp_step_count_by_client[0], 1)

    def test_prompt_prefix_keeps_private_tail_gradient(self):
        prompt = torch.nn.Parameter(
            torch.arange(12, dtype=torch.float32).view(6, 2) / 10
        )
        coefficients = torch.tensor(
            [[1.0, 0.5], [-0.25, 0.75], [0.4, -0.6]]
        )
        losses = torch.stack(
            [
                (prompt * coefficient).sum().square()
                for coefficient in coefficients
            ]
        )
        regular_gradient = torch.autograd.grad(
            losses.mean(),
            prompt,
            retain_graph=True,
        )[0]

        trainer = _trainer(losses.numel(), clip=0.5)
        torch.manual_seed(11)
        updates = noise_backend.compute_private_prompt_prefix(
            trainer=trainer,
            per_sample_losses=losses,
            parameter=prompt,
            block_size=2,
            shared_blocks=2,
        )
        losses.mean().backward()
        noise_backend.apply_private_gradients(updates)

        self.assertTrue(torch.equal(prompt.grad[4:], regular_gradient[4:]))
        self.assertFalse(torch.equal(prompt.grad[:4], regular_gradient[:4]))

    def test_all_trainers_use_the_per_sample_contract(self):
        trainers = (
            "FEDSEPT.py",
            "PROMPTFL.py",
            "FEDPGP.py",
            "FEDOTP.py",
            "FEDPHA.py",
            "PFEDMOAP.py",
            "DPFPL.py",
        )
        for trainer_file in trainers:
            source = (PROJECT_ROOT / "trainers" / trainer_file).read_text()
            self.assertIn("def _compute_private_updates(", source)
            self.assertIn("per_sample_losses", source)
            self.assertIn("noise_backend.apply_private_gradients", source)


if __name__ == "__main__":
    unittest.main()
