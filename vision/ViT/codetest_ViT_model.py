import unittest
import torch
from ViT_model import PatchEmbedding


class TestPatchEmbedding(unittest.TestCase):

    def setUp(self):
        self.patch_embedding = PatchEmbedding(patch_size=2, hidden_dim=3,
                                              image_height=4, image_width=4, n_channels=1, fake_init=True)

        self.image = torch.tensor([[0., 0., 1., 1.],
                                   [0., 0., 1., 1.],
                                   [2., 2., 3., 3.],
                                   [2., 2., 3., 3.]]).reshape(1, 1, 4, 4)

    def test_PatchEmbedding(self):
        x = self.patch_embedding(self.image)

        self.assertEqual(x[0, 0, :].tolist(), [0., 0., 0.])
        self.assertEqual(x[0, 1, :].tolist(), [4., 4., 4.])
        self.assertEqual(x[0, 2, :].tolist(), [8., 8., 8.])
        self.assertEqual(x[0, 3, :].tolist(), [12., 12., 12.])


if __name__ == '__main__':
    unittest.main()
