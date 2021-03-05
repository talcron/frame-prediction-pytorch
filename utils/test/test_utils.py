import unittest

import torch

from utils.functional import stats, frechet_inception_distance


class TestCov(unittest.TestCase):
    def test_cov_near_expected_for_vectors(self):
        cardinality = 5
        n_samples = 100_000
        mu = torch.rand(cardinality)
        sigma = torch.diag(torch.rand(cardinality)) * 100
        mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        data = mvn.sample_n(n_samples)

        _, measured_sigma = stats(data)
        self.assertTrue(torch.allclose(sigma, measured_sigma, atol=1.))


class TestFid(unittest.TestCase):
    def test_fid_for_same_dist_is_near_zero(self):
        cardinality = 5
        n_samples = 100_000
        mu = torch.rand(cardinality)
        sigma = torch.diag(torch.rand(cardinality)) * 100
        mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        data1 = mvn.sample_n(n_samples // 2)
        data2 = mvn.sample_n(n_samples // 2)

        fid = frechet_inception_distance(data1, data2)
        self.assertLess(fid, 1)

    def test_fid_for_different_dist_is_high(self):
        cardinality = 5
        n_samples = 100_000
        mu1 = torch.rand(cardinality)
        sigma1 = torch.diag(torch.rand(cardinality)) * 100
        mvn1 = torch.distributions.MultivariateNormal(loc=mu1, covariance_matrix=sigma1)
        data1 = mvn1.sample_n(n_samples // 2)
        mu2 = torch.rand(cardinality)
        sigma2 = torch.diag(torch.rand(cardinality)) * 100
        mvn2 = torch.distributions.MultivariateNormal(loc=mu2, covariance_matrix=sigma2)
        data2 = mvn2.sample_n(n_samples // 2)

        fid = frechet_inception_distance(data1, data2)
        self.assertGreater(fid, 10)




if __name__ == '__main__':
    unittest.main()
