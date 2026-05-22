import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import logging
from .kernels import *

from torch.distributions import biject_to

from .abstract_model import AbstractModel
from .pyro_extensions.guides import AutoSLPNormalReparamGuide

# Adapted from https://github.com/treigerm/sdvi_neurips
# replaces InverseGamma priors with Normal priors and transforms them to be LogNormal
# uses same kernel library as upix translated to torch
# reads data from relative path
# adds pell vs lppd eval

class GPKernelLearning(AbstractModel):
    does_lppd_evaluation = True
    slps_identified_by_discrete_samples = True

    input_dim = 1

    def __init__(self, data_path, jitter=1e-6):
        logging.info("gp_kernel_learning with custom kernels log transformed Normal priors")
        self.X, self.y, self.X_val, self.y_val = self.load_data(data_path)

        self.jitter = jitter

    @staticmethod
    def load_data(data_path):
        # support relative data path
        import os
        import pathlib
        data_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__)), "../../airline.csv")
        data = torch.tensor(np.loadtxt(data_path, delimiter=","))
        logging.info(f"{data_path=}")
        xs = data[:, 0]
        ys = data[:, 1]
        xs -= xs.min()
        xs /= xs.max()
        ys -= ys.mean()
        ys *= 4 / (ys.max() - ys.min())

        # Keep 10 % of data for validation.
        val_ix = round(xs.size(0) * 0.9)
        xs, xs_val = xs[:val_ix], xs[val_ix:]
        ys, ys_val = ys[:val_ix], ys[val_ix:]

        return xs, ys, xs_val, ys_val

    def sample_kernel_fn(self, kern_prefix: str) -> GPKernel:
        kernel_type = pyro.sample(
            f"{kern_prefix}kernel_type",
            dist.Categorical(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])),
            infer={"branching": True},
        )

        if kernel_type == 0.0:
            # Rational Quadratic kernel
            rq_kernel = UnitRationalQuadratic(
                transform_param("lengthscale", pyro.sample(f"{kern_prefix}lengthscale", dist.Normal(0.,1.0))),
                transform_param("scalemixture", pyro.sample(f"{kern_prefix}scalemixture", dist.Normal(0.,1.0)))
            )
            return rq_kernel
        elif kernel_type == 1.0:
            # Linear kernel
            linear_kernel = UnitPolynomialDegreeOne(
                transform_param("bias", pyro.sample(f"{kern_prefix}bias", dist.Normal(0.,1.0)))
            )
            return linear_kernel
        elif kernel_type == 2.0:
            # Squared Exponential kernel
            rbf_kernel = UnitSquaredExponential(
                transform_param("lengthscale", pyro.sample(f"{kern_prefix}lengthscale", dist.Normal(0.,1.0)))
            )
            return rbf_kernel
        elif kernel_type == 3.0:
            # Periodic
            periodic_kernel = UnitPeriodic(
                transform_param("lengthscale", pyro.sample(f"{kern_prefix}lengthscale", dist.Normal(0.,1.0))),
                transform_param("period", pyro.sample(f"{kern_prefix}period", dist.Normal(0.,1.0)))
            )
            return periodic_kernel
        elif kernel_type == 4.0:
            # Sum
            left_child = self.sample_kernel_fn(kern_prefix+"skern0.")
            right_child = self.sample_kernel_fn(kern_prefix+"skern1.")
            return Plus(left_child, right_child)
        elif kernel_type == 5.0:
            # Product
            left_child = self.sample_kernel_fn(kern_prefix+"tkern0.")
            right_child = self.sample_kernel_fn(kern_prefix+"tkern1.")
            return Times(left_child, right_child)
        else:
            raise ValueError(f"Unkown kernel type: {kernel_type}")

    def __call__(self):
        kernel = self.sample_kernel_fn(".")
        noise = transform_param("noise", pyro.sample("std", dist.Normal(0.,1.))) + 1e-5
        N = self.X.size(0)
        cov_matrix = kernel.eval_cov_vec(self.X) + noise * torch.eye(N)

        pyro.sample("obs", dist.MultivariateNormal(self.X.new_zeros(N), covariance_matrix=cov_matrix), obs=self.y)
        return kernel


    def make_parameter_plots(self, results, guide, branching_trace, file_prefix):
        if isinstance(guide, AutoSLPNormalReparamGuide):
            means = results["loc"]
            scale = [np.exp(v) for v in results["log_scale"]]
        else:
            logging.info(f"Parameter plotting for guide {guide} not supported.")
            return

        means = [
            [(site, v) for site, v in guide._unpack_latent(torch.tensor(cm))]
            for cm in means
        ]
        scale = [
            [(site, v) for site, v in guide._unpack_latent(torch.tensor(cs))]
            for cs in scale
        ]
        num_params = len(means[-1])

        # Plot final distributions
        fig, axs = plt.subplots(num_params, 1, figsize=(10, 4 * num_params))
        for ix in range(num_params):
            site = means[-1][ix][0]
            transform = biject_to(site["fn"].support)

            mean, std = means[-1][ix][1], scale[-1][ix][1]
            q_dist = dist.Normal(mean, std)
            xs = torch.linspace(mean - 3 * std, mean + 3 * std, 100)
            constrained_xs = transform(xs)
            log_densities = q_dist.log_prob(xs) + transform.inv.log_abs_det_jacobian(
                constrained_xs, xs
            )
            axs[ix].plot(constrained_xs, log_densities.exp())
            axs[ix].set_title(site["name"])

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_final_marginals.jpg")

        # Plot evolution of the means
        fig, axs = plt.subplots(num_params, 1, figsize=(10, 4 * num_params))
        for ix in range(num_params):
            site = means[0][ix][0]
            transform = biject_to(site["fn"].support)

            param_means = torch.tensor([x[ix][1] for x in means])
            constrained_param_means = transform(param_means)
            axs[ix].plot(constrained_param_means)
            axs[ix].set_title(f"{site['name']} mean")
            axs[ix].set_xlabel("Iteration")
            axs[ix].set_ylabel("Value")

        fig.tight_layout()
        fig.savefig(f"{file_prefix}_params.jpg")
        plt.close("all")

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        post_kernels = self.extract_posterior_kernels(posterior_samples)
        noises = [transform_param("noise", trace.nodes["std"]["value"]) for trace in posterior_samples]

        pell = torch.tensor(0.0) # this is what Reichelt et al. called LPPD but we call it posterior expected log-likelihood
        lppd = -torch.tensor(torch.inf)
        for kernel_fn, noise in zip(post_kernels, noises):
            lp = kernel_fn.posterior_predictive(self.X, self.y, noise, self.X_val, noise).log_prob(self.y_val).detach()
            pell += lp
            lppd = torch.logaddexp(lppd, lp)

        n = len(posterior_samples)
        return pell / n, lppd - torch.log(torch.tensor(n))

    def plot_posterior_samples(self, posterior_samples, fname):
        post_kernels = self.extract_posterior_kernels(posterior_samples)
        noises = [transform_param("noise", trace.nodes["std"]["value"]) for trace in posterior_samples]

        new_xs = torch.linspace(0, 1, 500)
        posterior_fs = torch.zeros((len(post_kernels), new_xs.size(0)))
        for ix in range(len(post_kernels)):
            with torch.no_grad():
                posterior_fs[ix, :] = (
                    post_kernels[ix].posterior_predictive(self.X, self.y, noises[ix], new_xs, noises[ix]).sample().detach()
                )

        f_post_mean = posterior_fs.mean(dim=0)
        f_post_std = posterior_fs.std(dim=0)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(new_xs, f_post_mean, label="Post mean", color="red")
        ax.fill_between(
            new_xs,
            f_post_mean - 2 * f_post_std,
            f_post_mean + 2 * f_post_std,
            color="red",
            alpha=0.2,
        )
        num_samples_to_plot = min(5, len(post_kernels))
        for ix in range(num_samples_to_plot):
            ax.plot(new_xs, posterior_fs[ix, :], color="green", alpha=0.3)

        ax.scatter(self.X, self.y, label="Data", color="black")
        ax.scatter(self.X_val, self.y_val, label="Held-out data")
        ax.set_xlim((-0.01, 1.01))
        ax.legend(loc="upper left")
        fig.savefig(fname)
        plt.close("all")

    @staticmethod
    def extract_posterior_kernels(posterior_samples) -> list[GPKernel]:
        post_kernels = [trace.nodes["_RETURN"]["value"] for trace in posterior_samples]
        return post_kernels

    @staticmethod
    def repr_samples(posterior_samples):
        return [repr(k) for k in GPKernelLearning.extract_posterior_kernels(posterior_samples)]